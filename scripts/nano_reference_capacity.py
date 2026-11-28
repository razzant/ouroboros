#!/usr/bin/env python3
"""Measure Nano's queued Web Main floor with pinned HF tokenizers, without generation.

Use --cache with the public inputs below; --fetch retrieves only missing pins.
This count-only fixture uses a synthetic drive and the real context renderer.
It neither connects to a model nor qualifies the serving tokenizer or window.
The report contains hashes and numbers, never prompt text. The pinned public
anti-slop text and a catalogue row are count-only fixture projections, not an
installed or admitted skill. Synthetic rules in the mature profile are initial
settings, not a Settings save; those two product paths need separate evidence.
W comes from exact start selection. F/B/F0 price the diagnostic floor; W0/WB
remove only the work or budgeted memory from W. The reply floor is a
reservation, not generated text or the prepared output cap.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
HUB_REVISION = "4bb28167e5537988e42456fdafc67f46b7c4f6b4"  # OuroborosHub #314
ANTI_SLOP = "anti-slop.SKILL.md"
ANTI_SLOP_BODY_SHA256 = "0c6baf3eccf43554b992c034340a2e2af937c6732a614293661f5c1bb99a8a4e"
MODELS = {
    "qwen35": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "", {
        "tokenizer.json": (12_807_982, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        "chat_template.jinja": (7_756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    }),
    "gemma4": ("unsloth/gemma-4-12b-it", "55cdba0740a9765956f49501f689a66b098feda3", "<bos>", {
        "tokenizer.json": (32_169_626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"),
        "chat_template.jinja": (18_924, "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b"),
    }),
}
SKILL_PIN = (7_563, "bbad0fc036bd8235e43f4527713e9a06d5c978392b6ed022444a133964396750")
# Fixed Qwen 3.5 reference measured by sequential prefix ablation (measure below), not recalibrated per PR:
# 6578 SYSTEM + 10520 BIBLE + 468 ARCHITECTURE + 406 DEVELOPMENT (resident chapter views).
GOVERNANCE_BASELINE = 17_972
GOVERNANCE_LIMIT = (GOVERNANCE_BASELINE * 105 + 99) // 100
REFERENCE_DEPS = {"tokenizers": "0.23.3", "jinja2": "3.1.6", "markupsafe": "3.0.3"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_pin(path, pin, url, fetch):
    size, digest = pin
    download = fetch and not path.exists()
    with (urlopen(url, timeout=45) if download else path.open("rb")) as source:
        data = source.read(size + 1)
    if len(data) != size or sha(data) != digest:
        raise ValueError(f"reference input {path.name}: expected {size} bytes, sha256 {digest}")
    if download:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def reference_counter(assets, bos):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    for package, required in REFERENCE_DEPS.items():
        if (installed := version(package)) != required:
            raise ValueError(f"expected {package}=={required}; found {installed}")

    def raise_exception(text):
        raise ValueError(text)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(assets["chat_template.jinja"].decode())
    tokenizer = Tokenizer.from_str(assets["tokenizer.json"].decode())
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def count(messages, tools, *, work=None, required=()):
        prompt = template.render(messages=messages, tools=tools, add_generation_prompt=True,
                                 enable_thinking=False, bos_token=bos, eos_token="")
        size = tokens(prompt)
        if any(prompt.count(span) != 1 for span in required):
            raise ValueError("the template did not retain an exact required input span")
        if work is None:
            return size
        text, occurrences = work
        if prompt.count(text) != occurrences or prompt.count(messages[-1]["content"]) != 1:
            raise ValueError("the template did not preserve the expected work occurrence and clock row")
        return size, sha(prompt.encode())

    return count, tokens


def reference_instruction(raw):
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

    manifest = parse_skill_manifest_text(raw.decode())
    if not (manifest.name == "anti-slop" and manifest.is_instruction()
            and manifest.always_in_context and manifest.version == "0.2.1" and manifest.body):
        raise ValueError("the pinned anti-slop file is not the expected always-on instruction")
    if sha(manifest.body.encode()) != ANTI_SLOP_BODY_SHA256:
        raise ValueError("the parsed anti-slop body differs from the pinned public instruction")
    return SimpleNamespace(name=manifest.name, location="ouroboroshub", manifest=manifest)


def instruction_projection(scope, instruction):
    """Supply fixed text at the two producer seams, before any Nano selection or count."""
    from ouroboros import context_sections, skill_catalogue
    from ouroboros.section_floor import HostSection
    from ouroboros.utils import truncate_review_artifact

    manifest = instruction.manifest
    live_section = skill_catalogue.always_in_context_section
    live_catalogue = context_sections._installed_skills_section
    fragment = skill_catalogue.always_in_context_fragment(instruction)

    def field(value):
        value = re.sub(r"\s+", " ", str(value or "")).strip().replace("|", "\\|").replace("#", "＃")
        return truncate_review_artifact(value, 260).replace("\n", " ")

    row = f"- {instruction.name} (instruction, v{manifest.version}, count-only fixture): {field(manifest.description)}"
    stanza = "\n".join((row, f"  Trigger: {field(manifest.when_to_use) or 'Not specified.'}",
        f"  Instructions: bucket={instruction.location!r}, skill_name={instruction.name!r}, path='SKILL.md'",
        "  Context: always; its text is in the Always-in-context skills section."))

    def section(skills):
        if getattr(skills, "error", None) or getattr(skills, "gaps", None) or any(
                skill.name == instruction.name for skill in skills):
            raise ValueError("reference instruction: conflicted or unreadable synthetic catalogue")
        original = live_section(skills)
        prefix = original or ("## Always-in-context skills\n\n"
                              "Pinned public instruction for this numerical fixture; admission not tested.")
        return prefix + "\n\n" + fragment

    def catalogue(*args, **kwargs):
        original = live_catalogue(*args, **kwargs)

        def add(text):
            head = "No skills are installed."
            if head in text:
                return text.replace(head, stanza, 1)
            return text + "\n" + stanza

        return HostSection(original.section_id, add(original.full_text), add(original.operative_view), original.reader)

    scope.enter_context(patch.object(skill_catalogue, "always_in_context_section", section))
    scope.enter_context(patch.object(context_sections, "_installed_skills_section", catalogue))
    return fragment, row


def check_instruction_views(views, instruction, *, present):
    from ouroboros.skill_catalogue import always_in_context_fragment

    heading = "## Always-in-context skills\n"
    title = f"### {instruction.name} (version {instruction.manifest.version})\n"
    fragment = always_in_context_fragment(instruction)
    expected = int(present)
    for label, rows in views.items():
        text = "\n\n".join(row["content"] for row in rows if row["role"] == "system")
        if text.count(instruction.manifest.body) != expected:
            raise ValueError(f"{label}: expected {expected} exact public anti-slop bodies")
        if any(text.count(marker) != expected for marker in (heading, title, fragment)) or (
                present and text.index(heading) >= text.index(fragment)):
            raise ValueError(f"{label}: the anti-slop section framing is incorrect")


def fixture_lines(label, size):
    return "\n".join(f"{label} {i:04d}: {sha(f'{label}/{i}'.encode())}" for i in range(size))


def mature_memory(memory):
    """Populate each of the six budgeted producers with fictitious, distinguishable bytes."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    root = memory.drive_root
    memory.ensure_files()
    values = {key: fixture_lines(f"Nano fixture {key}", rows) for key, rows in (
        ("identity", 65), ("scratchpad", 57), ("overview", 65), ("mark", 63), ("rules", 54),
    )}
    memory.identity_path().write_text(values["identity"], encoding="utf-8")
    block = memory.append_scratchpad_block(values["scratchpad"], source="fixture")
    if (memory.load_scratchpad_blocks() != [block] or not any(
            row.get("type") == "block_appended" and row.get("block") == block
            for row in map(json.loads, memory.journal_path().read_text(encoding="utf-8").splitlines()))):
        raise ValueError("could not seed the mature profile's canonical scratchpad and journal")

    def note(topic, text):
        write_knowledge_note(resolve_knowledge_address(root, topic, "global"), text, "overwrite", writer="turn")

    note("overview", values["overview"])
    for i in range(12):
        topic = f"nano-fixture-note-{i:02d}"
        summary = " ".join(sha(f"{topic}/{j}".encode()) for j in range(6))
        note(topic, f"---\nsummary: {summary}\n---\n# {topic}\n\nSynthetic note body.")
    store = ChronicleStore(root)
    store.ensure_activated()
    author = {"kind": "mind", "task_id": "fixture", "focus": {"role": "root", "task_id": "fixture"}}
    target = store.write_note(room_id="1", task_id="fixture", text="Synthetic mark target.", author=author)
    if not target.ok or not store.mark({"node_id": target.record["id"]}, values["mark"], author,
                                       room_id="1", scope="global").ok:
        raise ValueError("could not seed the mature profile's active mark")
    return values


def active_publication(memory):
    """Exact active bytes, not retained proposals or observational logs."""
    from ouroboros import config
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.knowledge import resolve_knowledge_address
    from ouroboros.nano_publication import PUBLICATION_REVISION, source_bytes

    root = memory.drive_root
    return {name: source_bytes(path) for name, path in {
        "identity": memory.identity_path(), "scratchpad": memory.scratchpad_path(),
        "scratchpad_blocks": memory.scratchpad_blocks_path(), "marks": ChronicleStore(root).log_path,
        "overview": resolve_knowledge_address(root, "overview", "global").path,
        "knowledge_index": resolve_knowledge_address(root, "nano-guarded-note", "global").path,
        "rules": config.SETTINGS_PATH, "publication": root / PUBLICATION_REVISION,
    }.items()}


def guarded_write(memory, markers, probe, *, label, operation, active_name):
    from ouroboros.nano_publication import MEMORY_CEILING

    before = active_publication(memory)
    counted = {}
    needle = markers[label]

    def observe(payload, key, size):
        if any(needle in row["content"] for row in payload["messages"]):
            if active_publication(memory)[active_name] != before[active_name]:
                raise ValueError(f"{label}: proposal was already active when the counter saw it")
            counted[key] = size

    probe["observe"] = observe
    try:
        result = str(operation())
    finally:
        probe.pop("observe")
    refusal = re.search(r"(?m)^(busy|unavailable|stale|incomplete|bootstrap_pending):", result)
    if refusal is not None:
        raise ValueError(f"guarded_writer.{label}: {refusal[1]}")
    if not counted:
        raise ValueError(f"guarded_writer.{label}: proposal did not reach the exact counter")
    after = active_publication(memory)
    if after["rules"] != before["rules"]:
        raise ValueError(f"guarded_writer.{label}: initial Settings bytes changed")
    observation = {"sha256": next(iter(counted)), "input_tokens": next(iter(counted.values()))}
    if "over_capacity:" in result:
        excess = re.search(r"over_capacity: memory excess ([0-9]+) tokens; work shortfall ([0-9]+) tokens", result)
        if before != after or excess is None or int(excess[1]) <= 0:
            raise ValueError(f"{label}: capacity refusal failed to preserve bytes/revision or prove memory pressure")
        return {"outcome": "over_capacity", "candidate_memory": MEMORY_CEILING + int(excess[1]),
                "work_shortfall": int(excess[2]), "active_sha256": publication_digests(after), "counted_proposal": observation}
    if (before["publication"] == after["publication"] or not after["publication"]
            or before[active_name] == after[active_name] or not after[active_name]
            or markers[label].encode() not in after[active_name]):
        raise ValueError(f"guarded_writer.{label}: counted, but active bytes or publication revision did not match")
    return {"outcome": "accepted", "active_sha256": publication_digests(after), "counted_proposal": observation}


def publication_digests(active):
    return {name: sha(raw) if raw is not None else None for name, raw in active.items()}


def publication_sample(memory, probe, rules, stage):
    from ouroboros.nano_host_publication import measure_publication_floor
    from ouroboros.nano_publication import NanoPublicationRefused

    rules_occurrences = {}

    def observe(payload, key, _size):
        rules_occurrences[key] = sum(row["content"].count(rules) for row in payload["messages"])

    probe["observe"] = observe
    try:
        sample = measure_publication_floor(memory.drive_root)
    except NanoPublicationRefused as exc:
        raise ValueError(f"guarded_profile.{stage}: {exc.reason}") from None
    finally:
        probe.pop("observe")
    if sample["status"] != "exact" or sample.get("knowledge_index_form") != "whole" or sample.get("world_form") != "whole":
        reason = sample.get("reason_code") or sample.get("reason") or "non_exact_or_shortened"
        raise ValueError(f"guarded_profile.{stage}: publication sample not exact/whole ({reason})")
    request_sha = sample["whole_index"]["native_input_sha256"]
    if rules_occurrences.get(request_sha) != 1 or rules_occurrences.get(
            sample["without_memory"]["native_input_sha256"]) != 0:
        raise ValueError(f"guarded_profile.{stage}: initial rules absent, duplicated or not isolated by ablation")
    return {"memory": sample["whole_memory_tokens"], "work_headroom": sample["minimum_work_headroom_at_reply_floor_tokens"],
            "background": sample["without_memory"]["input_tokens"], "reply": sample["reply_floor_tokens"],
            "request_sha256": request_sha}


def initialize_guarded_settings(drive, settings, rules):
    from ouroboros import config
    from ouroboros.safety_config import OWNER_RULES_KEY

    path = drive / "settings.json"
    if path != config.SETTINGS_PATH or drive != config.DATA_DIR:
        raise ValueError("guarded_profile.settings_bootstrap: unexpected profile binding")
    try:
        with path.open("xb") as output:
            output.write(config.serialize_settings({**settings, OWNER_RULES_KEY: rules}).encode("utf-8"))
        boot = config.load_settings(migrate=True, strict=True)
        saved = config.load_settings(strict=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"guarded_profile.settings_bootstrap: {type(exc).__name__}") from None
    if boot.get(OWNER_RULES_KEY) != rules or saved.get(OWNER_RULES_KEY) != rules:
        raise ValueError("guarded_profile.settings_bootstrap: initial rules failed ordinary readback")
    return sha(path.read_bytes())


def grow_guarded_memory(memory, probe, rules, settings_sha):
    """Reach the writer's actual threshold; let final F/W independently disagree with it."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address
    from ouroboros.nano_publication import MEMORY_CEILING
    from ouroboros.tools.chronicle import _chronicle_write, _memory_mark
    from ouroboros.tools.control_runtime import _update_identity, _update_scratchpad
    from ouroboros.tools.knowledge import _knowledge_write
    from ouroboros.tools.registry import ToolContext

    root = memory.drive_root
    ctx = ToolContext(ROOT, root, task_id="6e616e6f2d67726f", current_chat_id=1, current_task_type="task",
                      budget_drive_root=str(root))
    target = json.loads(_chronicle_write(ctx, kind="note", text="Synthetic guarded mark target."))
    if target.get("ok") is not True or ChronicleStore(root).get(target["node_id"]) is None:
        raise ValueError("guarded profile: no actual chronicle target")
    seed = {key: fixture_lines(f"Nano guarded {key}", 32) for key in ("identity", "scratchpad", "overview", "mark")}
    seed["knowledge_index"] = "Synthetic guarded index summary " + sha(b"Nano guarded index")
    markers = {key: value.splitlines()[0] for key, value in {**seed, "rules": rules}.items()}
    overview = resolve_knowledge_address(root, "overview", "global")
    initial = publication_sample(memory, probe, rules, "initial")
    original = read_knowledge_note(overview) if overview.path.is_file() else None

    def identity(text):
        return _update_identity(ctx, content=text, expected_revision=sha(memory.identity_path().read_bytes()))

    entries = (
        ("identity", lambda: identity(seed["identity"]), "identity"),
        ("scratchpad", lambda: _update_scratchpad(ctx, seed["scratchpad"]), "scratchpad_blocks"),
        ("overview", lambda: _knowledge_write(ctx, "overview", seed["overview"], mode="overwrite", scope="global",
                                               expected_revision=original.revision if original else None), "overview"),
        ("knowledge_index", lambda: _knowledge_write(ctx, "nano-guarded-note", "# Synthetic guarded note\n",
                                                      mode="overwrite", scope="global", summary=seed["knowledge_index"]), "knowledge_index"),
        ("mark", lambda: _memory_mark(ctx, node_id=target["node_id"], text=seed["mark"], scope="global"), "marks"),
    )
    accepted = {}
    for label, operation, active_name in entries:
        row = guarded_write(memory, markers, probe, label=label, operation=operation, active_name=active_name)
        if row["outcome"] != "accepted":
            raise ValueError(f"guarded profile: seed write {label} unexpectedly exceeded capacity")
        accepted[label] = {"revision_sha256": row["active_sha256"]["publication"], "counted_proposal": row["counted_proposal"]}
    seeded = publication_sample(memory, probe, rules, "seeded")
    lines = fixture_lines("Nano guarded identity", 640).splitlines()

    def try_identity(rows, *, rejected=False):
        body = "\n".join(lines[:rows - 1] + (["Nano guarded rejected identity: " + lines[rows - 1]]
                                              if rejected else [lines[rows - 1]]))
        markers["identity"] = body.splitlines()[-1]
        return guarded_write(memory, markers, probe, label="identity", operation=lambda: identity(body), active_name="identity")

    low, high = 32, len(lines)
    last_accept = accepted["identity"]
    if try_identity(high)["outcome"] != "over_capacity":
        raise ValueError("guarded profile: the fixed oversized proposal was not refused")
    while high - low > 1:
        trial = (high + low) // 2
        row = try_identity(trial)
        if row["outcome"] == "accepted":
            low = trial
            last_accept = {"revision_sha256": row["active_sha256"]["publication"], "counted_proposal": row["counted_proposal"]}
        else:
            high = trial
    markers["identity"] = lines[low - 1]
    boundary = publication_sample(memory, probe, rules, "boundary")
    final = publication_digests(active_publication(memory))
    if final["rules"] != settings_sha:
        raise ValueError("guarded_profile.boundary: initial Settings bytes changed")
    refusal = try_identity(high, rejected=True)
    rejected_marker = markers["identity"]
    markers["identity"] = lines[low - 1]
    if (refusal["outcome"] != "over_capacity" or refusal["active_sha256"] != final
            or not initial["memory"] < seeded["memory"] < boundary["memory"] <= MEMORY_CEILING
            or boundary["work_headroom"] < 15_000 or MEMORY_CEILING - boundary["memory"] > 256
            or last_accept["revision_sha256"] != final["publication"]):
        raise ValueError("guarded profile: no measured growth to the boundary followed by an unchanged refusal")
    return markers, rejected_marker, {"seeds": accepted, "last_accept": last_accept, "initial": initial,
                                      "rules_fixture": {"source": "initial_synthetic_settings", "owner_save": "not_run",
                                          "rules_sha256": sha(rules.encode()), "settings_sha256": settings_sha,
                                          "boundary_request_sha256": boundary["request_sha256"]},
                                      "seeded": seeded, "boundary": boundary,
                                      "accepted_rows": low, "refused_rows": high, "active_sha256": final, "refusal": refusal}


def working_text(counters, floor):
    lines = fixture_lines("Nano fixture work item", 1024).splitlines()
    start, end = 1, len(lines)
    while start < end:
        middle = (start + end) // 2
        text = "\n".join(lines[:middle])
        if min(tokens(text) for _count, tokens in counters.values()) >= floor + 32:
            end = middle
        else:
            start = middle + 1
    return "\n".join(lines[:start])


def remove_work(rows, text):
    """Ablate one exact span. Keep even the user row and its surrounding words."""
    if any(not isinstance(row["content"], str) for row in rows):
        raise ValueError("reference candidate did not flatten to text")
    hits = [(i, row) for i, row in enumerate(rows) if text in row["content"]]
    if len(hits) != 1 or hits[0][1]["role"] != "user" or hits[0][1]["content"].count(text) != 1:
        raise ValueError("the whole work span was duplicated, moved, lost or addressed")
    index, row = hits[0]
    return [*rows[:index], {**row, "content": row["content"].replace(text, "", 1)}, *rows[index + 1:]], index


def flat(blocks):
    return "\n\n".join(block["text"] for block in blocks if block["type"] == "text")


def without_mark_span(room, spans):
    matches = [span for span in set(spans) if span and span in room]
    if not matches:
        return room
    if len(matches) != 1 or room.count(matches[0]) != 1:
        raise ValueError("ambiguous marks in the selected room")
    span, = matches
    for part in (span + "\n\n", "\n\n" + span, span):
        if part in room:
            return room.replace(part, "", 1)
    raise AssertionError("mark span already checked")


def captured_root(raw, text, model, count, window, source, *, present, grown=False, guarded=False, drop_floor_body=False):
    from ouroboros import config, context, deployment_policy, local_model, memory_view, nano_host_publication
    from ouroboros import nano_minimum, section_floor, send_clock
    from ouroboros.agent import Env
    from ouroboros.llm import LLMClient
    from ouroboros.llm_helper_window import window_reply_candidate
    from ouroboros.local_model_server import input_fingerprint
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
    from ouroboros.memory import Memory
    from ouroboros.safety_config import OWNER_RULES_KEY
    from ouroboros.tool_policy import NANO_RESIDENT_SCHEMA_NAMES, initial_tool_schemas
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    drive = Path(os.environ["OUROBOROS_DATA_DIR"])
    if deployment_policy.current() is not None:
        raise ValueError("the reference run needs an isolated unmanaged profile")
    env = Env(repo_dir=ROOT, drive_root=drive)
    memory = Memory(drive_root=drive, repo_dir=ROOT)
    registry = ToolRegistry(ROOT, drive)
    tools = initial_tool_schemas(registry, "nano")
    names = [row["function"]["name"] for row in tools]
    if missing := NANO_RESIDENT_SCHEMA_NAMES - {"presence_finish", *names}:
        raise ValueError(f"ordinary Nano tool registry is incomplete: {sorted(missing)}")
    task = {"type": "task", "id": "nano-reference", "chat_id": 1, "source": "web", "use_local_model": True,
            "text": f"Use this synthetic work material:\n{text}\nEnd of synthetic work material."}
    when = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    if not source:
        from ouroboros.chronicle_store import ChronicleStore

        memory.ensure_files()
        ChronicleStore(drive).ensure_activated()
        source["memory"] = mature_memory(memory) if grown and not guarded else {}
        source["instruction"] = reference_instruction(raw)
        source["metadata"] = nano_host_publication.web_main_publication_metadata(drive, captured_at=when.isoformat())
    memory_fixture, instruction = source["memory"], source["instruction"]
    task["metadata"] = source["metadata"]
    ctx = ToolContext(repo_dir=ROOT, drive_root=drive, task_id=task["id"], current_chat_id=1,
                      current_task_type="task", task_metadata=task["metadata"], is_direct_chat=False)
    registry.set_context(ctx)
    settings = {"OUROBOROS_CONTEXT_MODE": "nano", "USE_LOCAL_MAIN": "1", "LOCAL_MODEL_CONTEXT_LENGTH": str(window)}
    port, pid = 8767, 31
    serving = {"context_window": window, "confirmed": True, "process_id": pid, "port": port}
    probes, selected_parts, mark_spans, governance = {}, {}, set(), {}

    def measure_input(payload, *, expected_port=None):
        if expected_port != port:
            raise ValueError("reference count escaped its synthetic local binding")
        size, rendered = count(payload["messages"], payload["tools"],
                               work=(text, sum(row["content"].count(text) for row in payload["messages"])))
        key = input_fingerprint(payload)
        if guarded and present and any(payload["messages"][0]["content"].count(span) != 1
                                       for span in instruction_spans):
            raise ValueError("guarded counter: public fixture differs from the exact-start projection")
        if observe := source.get("observe"):
            observe(payload, key, size)
        probes[(key, formatter_tool_sha(payload))] = (size, rendered)
        return {**serving, "supported": True, "input_is_exact": True, "input_tokens": size,
                "raw_context_window": window, "native_input_sha256": key, "output_limit_enforced": True,
                "reasoning_included_in_limit": True, "tokenizer_template_provenance": {
                    # This adapter satisfies the selector's test interface, not a serving claim.
                    "source": "serving_llama_formatter", "formatter": f"HF reference {model}",
                    "library_version": REFERENCE_DEPS["tokenizers"], "chat_format": model,
                    "launched_n_ctx": window, "rendered_prompt_sha256": rendered}}

    def without_body(views, core):
        if not drop_floor_body:
            return views
        part = next(section for section in core.semi_stable_sections if section.section_id == "always_on_skills")
        shown = views.get(part.section_id, part.full_text)
        return {**views, part.section_id: shown.replace(instruction.manifest.body, "", 1)}

    render_system = section_floor._system_blocks
    capture_system = section_floor.capture_system_projection
    render_marks = memory_view._marks_text

    def remember_marks(*args, **kwargs):
        span = render_marks(*args, **kwargs)
        mark_spans.add(span)
        return span

    def track(core, blocks, story, room, mode, addressed, receipts, whole, **kwargs):
        addressed = without_body(addressed, core)
        system, facts = capture_system(core, blocks, story, room, mode, addressed, receipts, whole, **kwargs)
        if mode == "nano":
            room_without_marks = without_mark_span(room, mark_spans)
            if grown and room_without_marks == room:
                raise ValueError("the mature selected view did not supply a marks producer")
            wb = render_system(core, blocks, story, room_without_marks, mode, addressed, whole,
                               omit=nano_minimum.MEMORY_SECTIONS)
            selected_parts[flat(system)] = (flat(wb), addressed)
        return system, facts

    def diagnostic(core, blocks, story, room, mode, addressed, whole, **kwargs):
        addressed = without_body(addressed, core)
        if "identity" in kwargs.get("omit", ()):
            for section in blocks[0]:
                label = getattr(section, "report_family", None) or section.section_id
                if label in {"system", "bible", "architecture_map", "development_map"}:
                    governance[label] = addressed.get(section.section_id, section.full_text)
        return render_system(core, blocks, story, room, mode, addressed, whole, **kwargs)

    manager = SimpleNamespace(serving_context_evidence=lambda **_kw: serving, measure_prepared_input=measure_input)
    evidence = SimpleNamespace(route_fp="nano-reference", status="confirmed", stale=False, window_tokens=window,
                               source="published_server_arguments", window_instance_id=str(pid))
    route = {"model": "local-model", "provider": "local", "use_local": True, "model_role": "main"}
    with (patch.object(config, "DATA_DIR", drive),
          patch.object(context, "utc_now_iso", return_value=when.isoformat()),
          patch("ouroboros.memory.utc_now_iso", return_value=when.isoformat()),
          patch.object(send_clock, "_now", return_value=when),
          patch.object(local_model, "local_dispatch_port", return_value=port),
          patch.object(local_model, "get_manager", return_value=manager),
          patch.dict(os.environ, {OWNER_RULES_KEY: memory_fixture.get("rules", "")}), ExitStack() as scope):
        instruction_spans = instruction_projection(scope, instruction) if present else ()
        if guarded:
            # Create the empty fixture with initial rules; this is not an owner action.
            scope.enter_context(patch.object(config, "REPO_DIR", ROOT))
            scope.enter_context(patch.object(config, "SETTINGS_PATH", drive / "settings.json"))
            rules = fixture_lines("Nano guarded rules", 32)
            settings_sha = initialize_guarded_settings(drive, {**settings, "USE_LOCAL_MAIN": "true",
                "LOCAL_MODEL_PORT": port, "LOCAL_MODEL_SOURCE": "synthetic-reference.gguf"}, rules)
            scope.enter_context(patch.object(LLMClient, "_get_local_client", side_effect=ValueError("generation forbidden")))
            stamp = send_clock.stamp_clock_note
            scope.enter_context(patch.object(send_clock, "stamp_clock_note",
                                            lambda payload, **_kw: stamp(payload, clock_at=when)))
            scope.enter_context(patch("ouroboros.utils.utc_now_iso", return_value=when.isoformat()))
            markers, rejected_marker, source["publication"] = grow_guarded_memory(memory, source, rules, settings_sha)
            if not probes:
                raise ValueError("the real memory writers did not reach the exact counter")
            probes.clear()  # writer observations cannot stand in for the following exact start
            scope.enter_context(nano_host_publication._as_next_root(nano_host_publication._read_current_settings()))
        else:
            scope.enter_context(patch.object(nano_host_publication, "_read_current_settings", return_value=settings))
        scope.enter_context(patch.object(section_floor, "capture_system_projection", track))
        scope.enter_context(patch.object(memory_view, "_marks_text", remember_marks))
        fit = {"tool_schemas": tools, "route_resolver": lambda *_a, **_kw: (route, evidence)}
        if "core" not in source:
            plan = context.build_context_fit_plan(env, memory, task, preferred_mode="nano", ctx=ctx,
                                                  tool_registry=registry, **fit)
            source["core"] = plan.core
        else:
            # The second tokenizer runs the same exact selection on the same logical capture.
            plan = context._build_context_fit_plan(env, source["core"], task, preferred_mode="nano", **fit)
        rows = plan.messages_for("nano")
        core = plan.core
        if plan.initial_mode != "nano" or plan.counter_status != "exact" or not probes:
            raise ValueError("no exact Nano start choice was observed; a preview is not evidence")
        if core.reference_book_errors or len(core.reference_books) != 2 or not core.base_prompt_source or not core.bible_source:
            raise ValueError("shipped SYSTEM/BIBLE/reference-book sources not captured")
        tools = plan.start_tool_schemas("nano")
        if not tools or [tool["function"]["name"] for tool in tools] != names:
            raise ValueError("the selected Nano schema frame differs from the resident schema frame")
        w, ceiling, prepare, _ = LLMClient(api_key="unused")._local_count_io(rows, tools, MAIN_LOOP_MAX_TOKENS,
                                                                            clock_at=when)
        w_key = (input_fingerprint(w), formatter_tool_sha(w))
        if w_key not in probes:
            raise ValueError("the prepared W was not counted by the start selector")
        if w["messages"][0]["content"] not in selected_parts:
            raise ValueError("selected W did not bind to a captured Nano projection")
        wb_system, addressed = selected_parts[w["messages"][0]["content"]]
        wb = {**w, "messages": [{**w["messages"][0], "content": wb_system}, *w["messages"][1:]]}
        with patch.object(nano_minimum, "_system_blocks", diagnostic):
            pair, forms, reason = nano_minimum._render(plan, rows, window=window, tool_names=names)
        if pair is None:
            raise ValueError(f"Nano root rendering unavailable: {reason}")
        n, b, f = (prepare(p) for p in (pair if len(pair) == 3 else (*pair, pair[0])))
        w_tokens, _ = probes[w_key]
        wire, short, _, _ = window_reply_candidate({**w, "max_tokens": ceiling}, "max_tokens",
            SimpleNamespace(profile="owner_nano"), window, {"input_tokens": w_tokens, "context_window": window},
            helper=False)
        if guarded and publication_digests(active_publication(memory)) != source["publication"]["active_sha256"]:
            raise ValueError("the subsequent fresh start did not preserve the accepted active revision")
    clock = {"role": "user", "content": send_clock.render_clock_note(send_clock.SendClockPolicy(), when)}
    if any(rows[-1:] != [clock] or rows.count(clock) != 1 for rows in (p["messages"] for p in (w, wb, n, b, f))):
        raise ValueError("reference inputs do not carry exactly one identical stamped clock row")
    sections = (*core.semi_stable_sections, *core.dynamic_head_sections, *core.dynamic_sections)
    index, = (part for part in sections if part.section_id == "knowledge_index")
    extensions, = (part for part in sections if part.section_id == "extension_tool_catalogue")
    installed, = (part for part in sections if part.section_id == "installed_skills")
    if not index.full_text or f["messages"][0]["content"].count(index.full_text) != 1:
        raise ValueError("F does not carry the exact full captured knowledge index")
    check_instruction_views({label: p["messages"] for label, p in (("W", w), ("N", n), ("B", b), ("F", f), ("WB", wb))},
                            instruction, present=present)
    # The fixed instruction row and the real extension catalogue are separate producers.
    catalogue_row = [line for line in installed.full_text.splitlines()
                     if line.startswith("- anti-slop (instruction, v0.2.1, count-only fixture):")]
    if catalogue_row != list(instruction_spans[1:]):
        raise ValueError("installed-skills projection did not retain the expected numerical fixture row")
    for label, payload in (("W", w), ("F", f), ("B", b), ("WB", wb)):
        system = payload["messages"][0]["content"]
        if system.count(extensions.operative_view) != 1 or system.count("Callable built-in and MCP tools,") != 1:
            raise ValueError(f"{label}: missing the ordinary Main extension reader and Connections form")
        if any(system.count(row) != 1 for row in catalogue_row):
            raise ValueError(f"{label}: missing the separately rendered installed anti-slop row")
    w_form = "outline" if "knowledge_index" in addressed else "whole"
    w_full_index = w["messages"][0]["content"].count(index.full_text) == 1
    if w_full_index != (w_form == "whole"):
        raise ValueError("the start's index form disagrees with the exact index bytes in W")
    if grown:
        if guarded:
            if any(sum(row["content"].count(rules) for row in payload["messages"]) != expected
                   for payload, expected in ((w, 1), (f, 1), (b, 0), (wb, 0))):
                raise ValueError("guarded fresh start: wrong occurrence count of the initial rules")
            if any(rejected_marker in row["content"] for payload in (w, f) for row in payload["messages"]):
                raise ValueError("the refused edit entered the fresh start")
        else:
            markers = {key: value.splitlines()[0] for key, value in memory_fixture.items()}
            markers["knowledge_index"] = "nano-fixture-note-00"
        for label, marker in markers.items():
            if (any(marker not in payload["messages"][0]["content"] for payload in (w, f))
                    or any(marker in row["content"] for payload in (b, wb) for row in payload["messages"])):
                raise ValueError(f"mature profile is missing the measured contribution of {label}")
    if guarded:
        source["publication"]["fresh_start_marker_sha256"] = {key: sha(value.encode()) for key, value in markers.items()}
    spans = [extensions.operative_view, "Callable built-in and MCP tools,", *catalogue_row]
    return {"W": w, "WB": wb, "F": f, "B": b, "F_forms": {**forms, "knowledge_index_form": "whole"},
            "W_forms": {"knowledge_index_form": w_form, "world_form": "reader" if "world" in addressed else "whole",
                        "extension_catalogue_form": "reader" if "extension_tool_catalogue" in addressed else "whole"},
            "spans": spans, "full_index": index.full_text, "W_full_index": w_full_index, "catalogue_row": catalogue_row,
            "core": core, "capture_sha256": plan.core_sha256, "governance": governance, "instruction_sha256": ANTI_SLOP_BODY_SHA256 if present else None,
            "probe": probes[w_key], "wire_cap": wire["max_tokens"], "short_reply": short, "caller_cap": ceiling,
            "publication": source.get("publication")}


def measure(count, rows, tools, governance):
    """Use sequential ablations so the charged components and remainder add back to the whole."""
    total = previous = count(rows, tools)
    part_tokens = {}
    for label in ("system", "bible", "architecture_map", "development_map"):
        part = governance.get(label, "")
        prefix = part + "\n\n"
        if not part or not rows[0]["content"].startswith(prefix):
            raise ValueError(f"could not bind {label} to the flattened Nano governance prefix")
        rows = [{**rows[0], "content": rows[0]["content"][len(prefix):]}, *rows[1:]]
        remaining = count(rows, tools)
        part_tokens[label] = previous - remaining
        previous = remaining
    rest = count(rows, [])
    # The template's joint tool price at the same prefix boundary as the rest of the decomposition.
    return total, part_tokens, previous - rest, rest


def problems(background, governance, memory, working, used, budgets, *, reference, mature):
    window, limit, work_floor, reply = budgets
    frame_limit = window - limit - work_floor - reply
    remaining_reserve = max(0, GOVERNANCE_LIMIT - governance) if reference else 0
    errors = []
    if reference and governance > GOVERNANCE_LIMIT:
        errors.append(f"shipped governance exceeds the reference growth allowance: {governance} > {GOVERNANCE_LIMIT}")
    if background + remaining_reserve > frame_limit:
        errors.append(f"frame plus unspent governance allowance exceeds the 80k budget: "
                      f"{background} + {remaining_reserve} > {frame_limit}")
    if not (limit * 3 // 4 if mature else 1) <= memory <= limit:
        errors.append(f"fixture memory W-WB does not exercise the intended reference budget: {memory}")
    if working < work_floor or used + reply > window:
        errors.append(f"full work witness does not fit: {working=} against {work_floor=}; {used=} + {reply=} >? {window=}")
    return errors


def joint_problems(f, b, f0, budgets):
    window, limit, work_floor, reply = budgets
    errors = []
    if not 0 <= f - b <= limit:
        errors.append(f"F-B memory exceeds the joint budget or is negative: {f - b} not in [0, {limit}]")
    if f - f0 < work_floor:
        errors.append(f"F-F0 work falls short of the joint floor: {f - f0} < {work_floor}")
    if f + reply > window:
        errors.append(f"F plus the reserved reply exceeds the window: {f} + {reply} > {window}")
    return errors


def json_sha(value):
    return sha(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def formatter_tool_sha(payload):
    """Supplement the native digest with the order visible to HF tool JSON rendering."""
    return json_sha(payload["tools"])


def profile_result(count, capture, text, budgets, *, reference, mature, grow):
    w, wb, f, b = (capture[name]["messages"] for name in ("W", "WB", "F", "B"))
    tools = capture["W"]["tools"]
    fixed = {k: v for k, v in capture["W"].items() if k != "messages"}
    if any({k: v for k, v in capture[name].items() if k != "messages"} != fixed for name in ("WB", "F", "B")):
        raise ValueError("reference forms differ outside the message cells")
    if not (w[1:] == wb[1:] and f[1:] == b[1:] and all(
            {**other[0], "content": ""} == {**original[0], "content": ""}
            and original[0]["role"] == "system" for original, other in ((w, wb), (f, b)))):
        raise ValueError("a memory ablation changed rows outside its system content")
    window, _, _, reply = budgets
    w0, index = remove_work(w, text)
    f0, f_index = remove_work(f, text)
    b0, b_index = remove_work(b, text)
    wb0, wb_index = remove_work(wb, text)
    if f_index != b_index or wb_index != index or index == len(w) - 1 or f_index == len(f) - 1:
        raise ValueError("an ablation would remove the clock or moved the work cell")
    catalogues = capture["spans"]
    with_index = [*catalogues, capture["full_index"]]
    chosen_spans = with_index if capture["W_full_index"] else catalogues
    w_tokens, w_rendered = count(w, tools, work=(text, 1), required=chosen_spans)
    if (w_tokens, w_rendered) != capture["probe"]:
        raise ValueError("W differs from the start selector's prepared witness")
    wb_tokens, wb_rendered = count(wb, tools, work=(text, 1), required=catalogues)
    w0_tokens, w0_rendered = count(w0, tools, work=(text, 0), required=chosen_spans)
    f_tokens, f_rendered = count(f, tools, work=(text, 1), required=with_index)
    f0_tokens, f0_rendered = count(f0, tools, work=(text, 0), required=with_index)
    background, b_rendered = count(b, tools, work=(text, 1), required=catalogues)

    def request_hash(rows):
        return json_sha({**fixed, "messages": rows})

    governance = dict(capture["governance"])
    if grow:
        original = governance["system"]
        grown = original + "\n\n" + fixture_lines("Nano governance growth control", 256)
        if not b0[0]["content"].startswith(original + "\n\n"):
            raise ValueError("negative-control source does not match the frame")
        b0 = [{**b0[0], "content": grown + b0[0]["content"][len(original):]}, *b0[1:]]
        governance["system"] = grown
    total, parts, tool_tokens, rest = measure(count, b0, tools, governance)
    charged = sum(parts.values())
    unspent = max(0, GOVERNANCE_LIMIT - charged) if reference else 0
    memory = w_tokens - wb_tokens
    marginal = w_tokens - w0_tokens
    errors = problems(total, charged, memory, marginal, w_tokens, budgets, reference=reference, mature=mature)
    if not capture["W_full_index"]:
        errors.append("exact start selected the index outline; the full index was not supplied")
    joint_errors = joint_problems(f_tokens, background, f0_tokens, budgets)
    wire_cap = capture["wire_cap"]
    if capture["short_reply"] or not reply <= wire_cap == min(capture["caller_cap"], window - w_tokens):
        errors.append(f"exact Nano reply allocator gives an out-of-window cap: {wire_cap}")
    f_sha, w_sha = request_hash(f), request_hash(w)
    core = capture["core"]
    return {"source_sha256": {"system": core.base_prompt_source[1], "bible": core.bible_source[1],
                              **{name: sha(value.encode()) for name, value in governance.items()
                                 if name.endswith("_map")}},
            "capture_sha256": capture["capture_sha256"], "F": f_tokens, "F0": f0_tokens, "B": background, "F_kind": "whole_knowledge_index",
            "forms_F": capture["F_forms"], "forms_W": capture["W_forms"], "W_matches_F_bytes": w_sha == f_sha,
            "W_is_shorter_than_F_in_tokens": w_tokens < f_tokens,
            "B0_frame": total, "WB0_frame": count(wb0, tools),
            "document_tokens_after_work_ablation": parts, "document_total": charged,
            "anti_slop_body_sha256": capture["instruction_sha256"],
            "anti_slop_fixture_catalogue_row_observed": bool(capture["catalogue_row"]),
            "anti_slop_fixture_catalogue_row_sha256": json_sha(capture["catalogue_row"]),
            "extension_catalogue_observed": True, "extension_reader_sha256": json_sha(catalogues[0]),
            "full_index_sha256": sha(capture["full_index"].encode()),
            "unspent_document_reserve_tokens": unspent,
            "uncommitted_frame_tokens": window - sum(budgets[1:]) - total - unspent,
            "tool_tokens_after_document_ablation": tool_tokens, "tool_schema_sha256": json_sha(tools),
            "tool_schema_names": [row["function"]["name"] for row in tools], "other_frame_tokens": rest,
            "whole_index_memory_minus_B": f_tokens - background, "memory_W_minus_WB": memory,
            "work_W_minus_W0": marginal, "reply_floor_remaining_after_W": window - w_tokens - reply,
            "negative_control": grow, "errors": errors + joint_errors,
            **({"guarded_publication": capture["publication"]} if capture.get("publication") is not None else {}),
            "joint_witness": {"work_F_minus_F0": f_tokens - f0_tokens,
                              "reply_floor_remaining_after_F": window - f_tokens - reply,
                              "F0_kind": "diagnostic_ablation_only", "changed_field": f"messages[{f_index}].content",
                              "work_occurrences_F_B_F0": [1, 1, 0], "clock_row_sha256": json_sha(f[-1]),
                              "F_sha256": f_sha, "B_sha256": request_hash(b), "F0_sha256": request_hash(f0),
                              "rendered_prompt_sha256": {"F": f_rendered, "B": b_rendered, "F0": f0_rendered},
                              "errors": joint_errors},
            "work_pair": {"W": w_tokens, "W0": w0_tokens, "WB": wb_tokens, "W0_kind": "diagnostic_ablation_only",
                          "WB_kind": "diagnostic_ablation_only", "changed_field": f"messages[{index}].content",
                          "memory_changed_field": "messages[0].content", "clock_row_sha256": json_sha(w[-1]),
                          "occurrences_in_W": 1, "occurrences_in_W0": 0, "occurrences_in_WB": 1,
                          "W_sha256": w_sha, "W0_sha256": request_hash(w0), "WB_sha256": request_hash(wb),
                          "rendered_prompt_sha256": {"W": w_rendered, "W0": w0_rendered, "WB": wb_rendered},
                          "fixed_fields_sha256": json_sha(fixed), "startup_probe_matches_W": True,
                          "prepared_reply_cap_tokens": fixed["max_tokens"], "caller_reply_cap_tokens": capture["caller_cap"],
                           "allocated_reply_cap_tokens": wire_cap, "short_reply": capture["short_reply"],
                          "work_sha256": sha(text.encode())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path, help="directory holding the pinned reference inputs")
    parser.add_argument("--fetch", action="store_true", help="GET missing pinned public inputs; default is offline")
    parser.add_argument("--negative-control", nargs="?", const="growth", choices=("growth", "missing-instruction"),
                        help="inject document growth (default control) or remove just the instruction body from B")
    args = parser.parse_args()
    try:
        counters = {}
        for label, (repo, revision, bos, pins) in MODELS.items():
            assets = {name: read_pin(args.cache / label / name, pin,
                                     f"https://huggingface.co/{repo}/resolve/{revision}/{name}", args.fetch)
                      for name, pin in pins.items()}
            counters[label] = reference_counter(assets, bos)
        instruction = read_pin(args.cache / ANTI_SLOP, SKILL_PIN,
                               f"https://raw.githubusercontent.com/razzant/OuroborosHub/{HUB_REVISION}/skills/anti-slop/SKILL.md",
                               args.fetch)
        # Same clean boundary used by safe_test; no installation HOME, data, or settings.
        from ouroboros.test_environment import isolated_environment
        from ouroboros.context_budget import _reply_floor
        from ouroboros.nano_publication import MEMORY_CEILING, QUALIFIED_WINDOW, WORK_FLOOR

        reply_floor = _reply_floor(65_536, True)[1]
        budgets = (QUALIFIED_WINDOW, MEMORY_CEILING, WORK_FLOOR, reply_floor)
        text = working_text(counters, WORK_FLOOR)

        blocked = []

        def no_connect(*_args, **_kwargs):
            blocked.append(True)
            raise OSError("the reference profile is offline")

        profiles = {}
        cases = [("fresh_bare", False, False), ("fresh_recommended", True, False)]
        if args.negative_control is None:
            cases += [("preseeded_overflow", True, True), ("mature_recommended", True, True)]
        for label, present, grown in cases:
            guarded = label == "mature_recommended"
            profiles[label] = {}
            groups = ({model: pair} for model, pair in counters.items()) if guarded else (counters,)
            for group in groups:
                with tempfile.TemporaryDirectory(prefix="nano-reference-") as scratch:
                    isolated = isolated_environment(Path(scratch), ROOT)
                    source = {}
                    with (patch.dict(os.environ, isolated, clear=True), patch.object(socket.socket, "connect", no_connect),
                          patch.object(socket.socket, "connect_ex", no_connect)):
                        for model, (count, _) in group.items():
                            capture = captured_root(instruction, text, model, count, QUALIFIED_WINDOW, source,
                                present=present, grown=grown, guarded=guarded,
                                drop_floor_body=present and args.negative_control == "missing-instruction")
                            profiles[label][model] = profile_result(count, capture, text, budgets, mature=grown,
                                reference=model == "qwen35", grow=present and args.negative_control == "growth")
        if blocked:
            raise ValueError("reference profile tried to connect while assembling its request")
        report = {"reference": "pinned HF templates and tokenizers / Nano queued Web Main fixture / no thinking",
                  "framing": {"add_generation_prompt": True, "enable_thinking": False, "eos_token": "",
                              "bos_token_by_model": {name: pin[2] for name, pin in MODELS.items()},
                              "add_special_tokens": False, "truncation": False, "padding": False,
                              "clock": "one fixed trailing user row", "reply": "reserved, not tokenized",
                              "work_occurrences_checked_in": ["prepared messages", "rendered template"]},
                  "instruction_fixture": "pinned public instruction body and count-only catalogue row, before ordinary selection",
                  "instruction_admission": "not_tested",
                  "start_reference_kind": "exact startup selector over pinned HF counts; synthetic queued Web Main, no serving formatter or request sent",
                  "models": {name: {"repo": repo, "revision": revision,
                                    "asset_sha256": {asset: pin[1] for asset, pin in pins.items()}}
                             for name, (repo, revision, _, pins) in MODELS.items()},
                  "anti_slop_hub_revision": HUB_REVISION, "anti_slop_file_sha256": SKILL_PIN[1],
                  "reference_dependencies": REFERENCE_DEPS, "frame_limit": QUALIFIED_WINDOW - sum(budgets[1:]),
                  "document_baseline": GOVERNANCE_BASELINE, "document_limit": GOVERNANCE_LIMIT,
                  "memory_work_reply_reserved": budgets[1:], "work_sha256": sha(text.encode()), "profiles": profiles}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        errors = [f"{label}/{model}: {error}" for label, results in profiles.items() if label != "preseeded_overflow"
                  for model, result in results.items() for error in result["errors"]]
        for model, row in profiles.get("preseeded_overflow", {}).items():
            if not (row["F"] - row["B"] > MEMORY_CEILING and row["F"] + reply_floor > QUALIFIED_WINDOW):
                errors.append(f"preseeded_overflow/{model}: the unchanged negative control did not exceed both limits")
        if errors:
            print("Nano 80k reference capacity failed:\n- " + "\n- ".join(errors), file=sys.stderr)
            return 1
        return 0
    except (ImportError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Nano reference capacity unavailable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
