"""Bind the Nano publication check to the next ordinary Main root, not the writer's turn.

Both server endpoints and workers read the current canonical profile. The worker's
possibly older task settings, current messages and room override are not authority
for an installation-wide change. Writers always take a fresh measurement. The
last finite qualification is a read model, not permission for future commits.
"""
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from ouroboros.nano_publication import (INSTRUCTION_ENABLE, INSTRUCTION_RELEASE, MEMORY_CEILING, PUBLICATION_REVISION, QUALIFIED_WINDOW, WORK_FLOOR,
                                        NanoPublication, NanoPublicationRefused, _checked_profile, _marker_field,
                                        _retain_proposal, local_root_publication, shown_digest, source_bytes)

_ROOT_ID = "6e616e6f2d666c6f"
# ``start_server`` passes ``--chat_format <name>`` only when Settings name one.
_CHAT_FORMAT_ARGV_RE = re.compile(r"\s--chat_format (\S+)")
# The saved launch values that decide which formatter a fresh Main meets; the GPU split does not.
_LAUNCH_KEYS = ("LOCAL_MODEL_SOURCE", "LOCAL_MODEL_FILENAME", "LOCAL_MODEL_PORT", "LOCAL_MODEL_CONTEXT_LENGTH",
                "LOCAL_MODEL_CHAT_FORMAT")


def _read_current_settings():
    from ouroboros import config
    from ouroboros.server_runtime import apply_runtime_provider_defaults
    from ouroboros.settings_integrity import (SettingsDocumentUnknown, SettingsIntegrityError,
                                              read_settings_json_strict, settings_read_environ, skill_environ)

    try:
        # A normal strict reader can wait behind the Settings writer after a
        # failed read. We also call this from that writer's commit boundary.
        raw = read_settings_json_strict(config.SETTINGS_PATH)
        with settings_read_environ(skill_environ()):
            settings, _, _ = apply_runtime_provider_defaults(config.settings_of_document(raw, present=raw is not None))
            return settings
    except (SettingsDocumentUnknown, SettingsIntegrityError) as exc:
        raise NanoPublicationRefused("unavailable", "the active Settings profile could not be read") from exc


def _qualified():
    """``(settings, serving)`` while the installation's Main is Nano on the local 80k model, else ``None``."""
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.local_model import get_manager, local_dispatch_port, local_model_settings
    from ouroboros.model_slots import MODEL_CONTEXT_WINDOWS_KEY, model_role_option

    settings = _read_current_settings()
    if settings.get("OUROBOROS_CONTEXT_MODE") != "nano" or not _active_main_route(settings)["use_local"]:
        return None
    launch = local_model_settings(settings)
    role_window = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "main", settings=settings) or 0)
    serving = get_manager().serving_context_evidence(bind_endpoint=True)
    if serving.get("confirmed") and serving.get("port") == local_dispatch_port():
        window = serving.get("context_window")
    else:
        window = launch["LOCAL_MODEL_CONTEXT_LENGTH"]
        # The requested launch can be a candidate before it serves. It cannot
        # borrow the formatter already listening on a different dispatch port.
        serving = {}
    if window != QUALIFIED_WINDOW or role_window not in (0, QUALIFIED_WINDOW):
        return None
    return settings, serving


def _current_profile():
    found = _qualified()
    return found[0] if found is not None else None


def main_extension_tool_reader(root, task, mode):
    """Only this installation's configured 80k Nano Main has the alternate names view."""
    from ouroboros.config import DATA_DIR

    if (mode != 'nano' or Path(root).resolve() != Path(DATA_DIR).resolve()
            or task.get('chat_id') != 1 or task.get('parent_task_id') or task.get('_presence_turn')
            or task.get('delegation_role') not in (None, '', 'root')):
        return False
    try:
        found = _qualified()
        return found is not None and found[1].get('confirmed') is True
    except (OSError, ValueError, NanoPublicationRefused):
        return False


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _saved_launch(settings) -> dict:
    """The saved launch values a Start applies; never credentials, rules or task settings."""
    from ouroboros.local_model import local_model_settings

    values = local_model_settings(settings)
    return {key: values[key] for key in _LAUNCH_KEYS}


def _launch_of(command: str):
    """The counting launch a live server's own command line names, or ``None`` for another program."""
    from ouroboros.local_model import _SERVING_CONTEXT_ARGV_RE, _SERVING_MODEL_ARGV_RE, local_artifact_key

    if "ouroboros.local_model_server" not in command:
        return None
    model, window = _SERVING_MODEL_ARGV_RE.search(command), _SERVING_CONTEXT_ARGV_RE.search(command)
    if model is None or window is None:
        return None
    chat = _CHAT_FORMAT_ARGV_RE.search(command)
    path = model.group(1).strip('"')
    key = local_artifact_key(model_path=path)
    # A routing key identifies a family; a checked formatter belongs to a specific HF snapshot.
    snapshot = {"snapshot_path": os.path.normcase(os.path.abspath(path))} if key.startswith("hf:") else {}
    return {"model": key, **snapshot, "port": int(model.group(2)),
            "chat_format": chat.group(1).strip('"') if chat else "", "context": int(window.group(1))}


def _served_launch(serving):
    """The live server's launch, read from its own command line through its published binding.

    The server and every worker read the same argv on every platform (``platform_layer.process_command``),
    so a fresh Main's formatter is named by what serves now, never by Settings saved but not yet
    applied. ``None`` where no binding names the serving process or its argv cannot be read (no
    ``psutil`` on Windows, a failed ``ps``): the owner's report of unapplied values then decides.
    """
    from ouroboros import config
    from ouroboros.platform_layer import process_command
    from ouroboros.server_process import read_service_bindings

    if serving.get("confirmed") is not True:
        return None
    try:
        binding = read_service_bindings(Path(config.DATA_DIR)).get("local_model") or {}
    except (OSError, ValueError) as exc:
        raise NanoPublicationRefused("unavailable", "the local model's service binding could not be read") from exc
    if (type(binding.get("pid")) is not int or binding["pid"] != serving.get("process_id")
            or binding.get("port") != serving.get("port")):
        return None
    return _launch_of(process_command(binding["pid"]))


def _names(saved: dict, served: dict) -> bool:
    """Whether the saved launch values start exactly the serving launch again, decided offline."""
    if (saved["LOCAL_MODEL_PORT"], saved["LOCAL_MODEL_CHAT_FORMAT"], saved["LOCAL_MODEL_CONTEXT_LENGTH"]) != (
            served["port"], served["chat_format"], served["context"]):
        return False
    source, model = saved["LOCAL_MODEL_SOURCE"], served["model"]
    filename = saved["LOCAL_MODEL_FILENAME"].replace("\\", "/")
    if not source:
        return False
    if not model.startswith("hf:"):
        try:
            return len({os.path.normcase(os.path.realpath(os.path.expanduser(path))) for path in (source, model)}) == 1
        except (OSError, ValueError):
            return False
    # The downloader resolves a bare file name to its one match in the repository.
    return model == f"hf:{source}:{filename}" or (bool(filename) and "/" not in filename
                                                   and model.startswith(f"hf:{source}:") and model.endswith("/" + filename))


def _launch_pending(settings) -> bool:
    """The owned server reports saved launch values it has not applied (only its owner process knows)."""
    from ouroboros.local_model import get_manager

    application = getattr(get_manager(), "settings_application", None)
    if application is None:
        return False
    try:
        report = application(settings)
    except Exception:  # no report is no proof that the saved values serve
        return True
    return bool((set(report.get("pending_keys") or ()) | set(report.get("unknown_keys") or ())) & set(_LAUNCH_KEYS))


def _profile_key() -> str | None:
    """The counting capability a fresh Main meets: the serving launch, else the saved launch values.

    While a server serves, its own launch names the profile, so Settings saved but not applied
    never inherit its confirmation. While none serves, saved values proven to start the last
    checked launch still name it: a stopped server is an outage of a known formatter, not a new
    profile. Where no binding names the serving process, the saved values name it unless its
    owner reports them not applied.
    """
    from ouroboros import config

    found = _qualified()
    if found is None:
        return None
    settings, serving = found
    if (served := _served_launch(serving)) is not None:
        return "launch:" + _digest(served)
    saved = _digest(_saved_launch(settings))
    if serving.get("confirmed") is True and _launch_pending(settings):
        return "pending:" + saved
    marker = source_bytes(Path(config.DATA_DIR).resolve() / PUBLICATION_REVISION)
    if (checked := _checked_profile(marker)) and _marker_field(marker, "checked_saved") == saved:
        return checked
    return "settings:" + saved


def profile_stamp() -> dict:
    """The saved launch values proven to start the serving launch again, recorded beside its confirmation."""
    found = _qualified()
    if found is None:
        return {}
    settings, serving = found
    served, saved = _served_launch(serving), _saved_launch(settings)
    return {"checked_saved": _digest(saved)} if served is not None and _names(saved, served) else {}


def current_rules_digest():
    """Fingerprint the rules a fresh Main would read, without returning their text."""
    from ouroboros.nano_minimum import rules_digest
    from ouroboros.safety_config import owner_rule_sections

    with _as_next_root(_read_current_settings()):
        return rules_digest(owner_rule_sections())


def serving_counter_key(floor):
    """A prepared count proves a capability only on the current dispatch binding."""
    from ouroboros.local_model import get_manager, local_dispatch_port

    if floor.get("context_window") != QUALIFIED_WINDOW or floor.get("port") != local_dispatch_port():
        return None
    live = get_manager().serving_context_evidence(bind_endpoint=True)
    if (live.get("confirmed") is not True or any(floor.get(k) != live.get(k)
            for k in ("process_id", "port", "context_window"))):
        return None
    return _profile_key()


@contextmanager
def _as_next_root(settings, *, rule_override=None):
    import os

    from ouroboros.config import apply_settings_to_env
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    projected = dict(os.environ)
    apply_settings_to_env(settings, environ=projected)
    if rule_override is not None:
        from ouroboros.safety_config import OWNER_RULES_KEY

        projected[OWNER_RULES_KEY] = rule_override  # prospective owner authoring, including first save and clearing
    with task_settings_scope(task_settings_snapshot(settings, projected)):
        yield


@contextmanager
def _proposed_rules(settings, _root, changes):
    from ouroboros.safety_config import OWNER_RULES_KEY

    rule = next(change for change in changes if change.subject == "owner_rules")
    raw = str(settings.get(OWNER_RULES_KEY) or "").encode("utf-8")
    if raw != rule.before:
        raise NanoPublicationRefused("stale", "the saved rules changed before their count")
    proposed = rule.after.decode("utf-8")
    with _as_next_root({**settings, OWNER_RULES_KEY: proposed}, rule_override=proposed):
        yield


def _first_start_view(root: Path, repo: Path):
    """Memory as the next fresh Main reads it after its own first-start initialization, read without writing.

    Only text every start writes the same way is priced here: an absent identity reads as its constant
    default. An absent scratchpad or WORLD is written by the start itself (a time stamp; the host's
    measured block with its working directory, PATH and date), so this process cannot know those bytes:
    the edit waits for the ordinary start that writes them (``bootstrap_pending``: the proposal is kept,
    the active version stays, the counter is not suspected), and no probe of the host runs for it. A file
    that exists but cannot be read or parsed refuses as an incomplete read; it is never an absence.
    Nothing here creates a memory or WORLD file, so a preview, a refused write and a Reset leave none.
    """
    from ouroboros.memory import Memory
    from ouroboros.nano_publication import prospective_text
    from ouroboros.utils import read_text

    def started(path: Path) -> NanoPublicationRefused:
        return NanoPublicationRefused("bootstrap_pending", f"{path.name} is written by the ordinary start in its own "
                                      "environment; this edit can be measured once it exists")

    def absent(path: Path) -> bool:
        try:
            path.stat()
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise NanoPublicationRefused("incomplete", f"{path.name} could not be read: {type(exc).__name__}") from exc
        return False

    def whole(read, path: Path):
        try:
            return read(path)
        except (OSError, UnicodeError) as exc:
            raise NanoPublicationRefused("incomplete", f"{path.name} could not be read: {type(exc).__name__}") from exc

    class FirstStartView(Memory):
        renders_world_preview = True

        def ensure_files(self) -> None:
            return None

        def load_identity(self) -> str:
            path = self.identity_path()
            return self._default_identity() if absent(path) else whole(lambda item: item.read_bytes().decode("utf-8"), path)

        def load_scratchpad(self) -> str:
            path = self.scratchpad_path()
            if absent(path):
                raise started(path)
            return whole(read_text, path)

        def read_scratchpad_blocks(self):
            path = self.scratchpad_blocks_path()
            if absent(path):
                return [], True  # the ordinary renderer reads the known legacy Markdown instead

            def unreadable(exc):
                raise NanoPublicationRefused("incomplete", f"{path.name} could not be read: {type(exc).__name__}") from exc

            return super().read_scratchpad_blocks(create_lock=False, on_unavailable=unreadable)

        def scratchpad_journal_visible(self) -> bool:
            # ensure_files() creates this journal; assembly reads its address, not its contents.
            absent(self.journal_path())  # a stat error is not known absence
            return True

        def load_world_profile(self) -> str:
            path = self.world_path()
            if (proposed := prospective_text(path)) is not None:
                return proposed.decode("utf-8")  # the map writer's own canonical bytes, measured block kept
            if absent(path):
                raise started(path)
            return whole(lambda item: item.read_bytes().decode("utf-8"), path)

    return FirstStartView(drive_root=root, repo_dir=repo)


def web_main_publication_metadata(root: Path, *, captured_at: str) -> dict:
    """A sized example of a Web Main decision frame, not a real message or a worst-case input bound."""
    from types import SimpleNamespace

    from ouroboros.client_surface import normalize_client_surface
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.server_routing_context import RootSnapshot, _decision_turn_metadata, addressable_roots

    snapshot = load_queue_document(root)
    running = {str(row.get("id") or row["task_id"]): row for row in (snapshot.get("running") or ())
               if isinstance(row, dict) and (row.get("id") or row.get("task_id"))}
    pending = [{**(row['task'] if isinstance(row.get('task'), dict) else {}), **row}
               for row in (snapshot.get("pending") or ()) if isinstance(row, dict)]
    router = SimpleNamespace(DRIVE_ROOT=root, RUNNING=running, PENDING=pending)
    roots = (RootSnapshot(gap="queue_snapshot_unreadable") if snapshot.get("_snapshot_invalid") else
             addressable_roots(router))
    surface = normalize_client_surface({"ua": "Synthetic Web Main sizing probe", "pywebview": False,
        "narrow_layout": False, "coarse_pointer": False, "viewport": {"w": 1440, "h": 900},
        "interface_size": 100, "device_pixel_ratio": 1, "captured_at": captured_at, "timezone": "UTC"})
    return _decision_turn_metadata(router, 1, "nano-publication-probe", {"source": "web", "origin_suppressed": True,
        "client_surface": {**surface, "received_at": captured_at}}, roots, decided_project_id="")


def _root_counts(root, settings, changes):
    from functools import partial

    from ouroboros.agent import Env
    from ouroboros.chronicle_publication import prospective_chronicle
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.config import REPO_DIR
    from ouroboros.knowledge_publication import intermediate_projections, prospective_knowledge
    from ouroboros.llm import LLMClient
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
    from ouroboros.nano_activation import pending_instruction_rollback_view, prospective_instruction
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.utils import utc_now_iso

    chronicle = ChronicleStore(root)
    try:
        chronicle.log_path.stat()
    except FileNotFoundError as exc:
        raise NanoPublicationRefused("bootstrap_pending", "the ordinary start must activate the chronicle first") from exc
    except OSError as exc:
        raise NanoPublicationRefused("incomplete", "the chronicle journal could not be read") from exc
    if not chronicle.activation():
        raise NanoPublicationRefused("bootstrap_pending", "the ordinary start must activate the chronicle first")
    repo = Path(REPO_DIR)
    env, memory = Env(repo, root), _first_start_view(root, repo)
    metadata = web_main_publication_metadata(root, captured_at=utc_now_iso())
    task = {"type": "task", "id": _ROOT_ID, "chat_id": 1, "delegation_role": "root",
            "budget_drive_root": str(root), "metadata": metadata}
    ctx = ToolContext(repo, root, current_chat_id=1, current_task_type="task", task_id=_ROOT_ID,
                      task_metadata=metadata)
    tools = ToolRegistry(repo, root)
    tools.set_context(ctx)
    overlays = {"knowledge_note": prospective_knowledge, "knowledge_claims": prospective_knowledge,
                "chronicle": prospective_chronicle, "owner_rules": partial(_proposed_rules, settings),
                INSTRUCTION_ENABLE: prospective_instruction, INSTRUCTION_RELEASE: prospective_instruction}
    with _as_next_root(settings):
        envelope = local_root_publication(env, memory, task, author_ctx=ctx, root_ctx=ctx, tool_registry=tools,
            client=LLMClient(), caller_max=MAIN_LOOP_MAX_TOKENS, overlays=overlays,
            tool_schemas=initial_tool_schemas(tools, context_mode="nano"), scope_revision=lambda: None,
            intermediate=intermediate_projections, alternate_view=pending_instruction_rollback_view)
        return envelope.capture(changes)


def _file_stamp(path: Path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


def _settings_framing() -> str:
    """The saved Settings besides the owner's rules (they are counted memory), as one digest."""
    from ouroboros import config
    from ouroboros.settings_integrity import SettingsDocumentUnknown, SettingsIntegrityError, read_settings_json_strict

    try:
        return _framing_of(read_settings_json_strict(config.SETTINGS_PATH))
    except (SettingsDocumentUnknown, SettingsIntegrityError) as exc:
        raise NanoPublicationRefused("unavailable", "the active Settings profile could not be read") from exc


def _framing_of(document) -> str:
    """The document over its defaults and without the rules: writing a default explicitly changes nothing, and
    neither the process environment nor the reader's process does. Non-secret values enter as ``setting_tokens``
    hashes them, a secret only as its presence."""
    from ouroboros import config
    from ouroboros.gateway.owner_settings import setting_tokens
    from ouroboros.safety_config import OWNER_RULES_KEY
    from ouroboros.settings_scales import defaults_for_settings_document, normalize_effort_document

    try:
        raw = config.normalize_settings_raw(document or {})
        saved = {**defaults_for_settings_document(document is not None), **raw, **normalize_effort_document(raw)}
        keys = tuple(sorted(key for key in saved if key != OWNER_RULES_KEY))
        tokens = setting_tokens(saved, keys)
        return _digest([tokens, [key for key in keys if key not in tokens and saved.get(key)]])
    except TypeError as exc:
        raise NanoPublicationRefused("unavailable", "the saved Settings are not plain data") from exc


def _active_sources(root: Path):
    """Sources a fresh Main reads whose loss withdraws an observation; presence and readability only."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.knowledge import OVERVIEW_TOPIC, resolve_knowledge_address
    from ouroboros.memory import Memory

    memory = Memory(drive_root=root)
    return (memory.identity_path(), memory.scratchpad_path(), memory.scratchpad_blocks_path(), memory.journal_path(),
            memory.identity_journal_path(), ChronicleStore(root).log_path,
            resolve_knowledge_address(root, OVERVIEW_TOPIC, "global").path)


def _source_state(path: Path):
    """``None`` for a proven absence, else whether this process may read it; any other stat error raises."""
    try:
        path.stat()
    except FileNotFoundError:
        return None
    return os.access(path, os.R_OK)


def _profile_witness(root: Path):
    """Fixed bindings that can invalidate the observation without a counter call.

    A known absence is a value here; an I/O error raises and is never read as absence. A profile
    in an owner Reset, or whose launch witness cannot be proven, is not the one a root counted.
    """
    from ouroboros import config, deployment_policy
    from ouroboros.launch_roots import RESETTING, read_witness
    from ouroboros.local_model import local_dispatch_port
    from ouroboros.memory import Memory
    from ouroboros.provider_connections import settings_snapshot
    from ouroboros.server_process import read_service_bindings

    found = _qualified()
    if found is None:
        return None
    _settings, serving = found
    memory = Memory(drive_root=root)
    if (serving.get("confirmed") is not True or serving.get("context_window") != QUALIFIED_WINDOW
            or not serving.get("process_id") or serving.get("port") != local_dispatch_port()):
        return None
    if read_witness(root)[0] in (RESETTING, "unreadable", "invalid"):
        raise NanoPublicationRefused("unavailable", "the profile is in an owner Reset or its launch witness is unproven")
    try:
        policy = deployment_policy.current()
    except deployment_policy.DeploymentPolicyError as exc:
        raise NanoPublicationRefused("unavailable", "the deployment rules could not be read") from exc
    binding = read_service_bindings(root).get("local_model")
    repo = Path(config.REPO_DIR)
    fixed_sources = (repo / "BIBLE.md", repo / "prompts/SYSTEM.md", repo / "docs/DESIGN.md",
                     repo / "docs/ARCHITECTURE.md", repo / "docs/DEVELOPMENT.md", memory.world_path(),
                     root / "memory/registry.md", root / "state/server_process.json")
    # Keep only digests of the effective policy, the shown Settings and the service binding; no
    # policy text, owner input, model path, or Settings value in the observation.
    # The skill catalogue is not discovered here: this runs under the exclusive fence, and every
    # catalogue transition counts itself or takes ``publication_fence``, which withdraws the observation.
    # The second element is what a rules-only save leaves unchanged; a save that also changes any other
    # setting records no observation (``_PublicationTicket._record_protection``).
    return (settings_snapshot(), _settings_framing(), serving["process_id"], serving["port"],
            sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
            policy.digest if policy else None, tuple(_file_stamp(path) for path in fixed_sources),
            tuple(sha256(raw).hexdigest() if (raw := source_bytes(path)) is not None else None
                  for path in (memory.world_path(), root / "memory/registry.md")),
            # Presence and readability only: checked writers replace these files, a first start creates them.
            tuple(_source_state(path) for path in _active_sources(root)))


def current_protection(root: Path) -> dict:
    """Describe the measured Main envelope while its fixed inputs still match.

    Both skill texts a fresh root shows are compared as rendered, by their own renderers, with the
    texts the observation counted: the always-in-context block and the Installed Skills catalogue.
    An observation recorded without the catalogue's digest proves nothing about it and is not shown.
    """
    from types import SimpleNamespace

    from ouroboros.config import DATA_DIR
    from ouroboros.context import _world_section
    from ouroboros.context_sections import _build_installed_skills_section
    from ouroboros.memory import Memory
    from ouroboros.skill_catalogue import always_in_context_section, discover_context_skills

    root = Path(root).resolve()
    if root != Path(DATA_DIR).resolve():
        return {}
    marker = root / PUBLICATION_REVISION
    try:
        with marker.open("rb") as handle:
            raw = handle.read(4097)
        doc = json.loads(raw) if len(raw) <= 4096 else None
        if not isinstance(doc, dict):
            return {}
        key = _profile_key()
        if key is None or key != doc.get("checked_profile"):
            return {}
        witness = _profile_witness(root)
        if witness is None or doc.get("reason") == "confirmed_counter_unavailable":
            return {"reason": "confirmed_counter_unavailable"}
        proof = doc.get("proof")
        if not isinstance(proof, dict) or proof.get("world_form") != "whole":
            return {}
        b, reply = proof.get("background_tokens"), proof.get("reply_floor_tokens")
        if type(b) is not int or type(reply) is not int or min(b, reply) <= 0:
            return {}
        if b + reply + MEMORY_CEILING + WORK_FLOOR > QUALIFIED_WINDOW:
            return {}
        world_raw = source_bytes(Memory(drive_root=root).world_path())
        world_text = world_raw.decode("utf-8") if world_raw is not None else ""
        if ("world_digest" in proof
                and proof["world_digest"] != shown_digest(_world_section(world_text) if world_text.strip() else "")):
            return {}
        skills = discover_context_skills(root)
        if getattr(skills, "error", None) or getattr(skills, "gaps", None):
            return {}
        # Both forms of the catalogue and the whole always-in-context block, as ``RootFloor`` counted them.
        env = SimpleNamespace(drive_root=root)
        plain = _build_installed_skills_section(env, skills=skills)
        discoverable = _build_installed_skills_section(env, skills=skills, separate_extension_tools=True)
        if (proof.get("instruction_digest") != shown_digest(always_in_context_section(skills))
                or proof.get("catalogue_digest") not in {shown_digest(plain), shown_digest(discoverable)}
                or doc.get("profile") != repr(witness) or marker.read_bytes() != raw):
            return {}
        return {"status": "protecting", "reason": "checked_controlled_growth",
                "protection_scope": "unchanged_main_background", "protection_observed_at": doc["observed_at"],
                "qualified_background_tokens": b, "qualified_reply_floor_tokens": reply}
    except (OSError, ValueError, KeyError, NanoPublicationRefused):
        return {}


def host_admission(root: Path, *, author=None, retain=None):
    from ouroboros import config

    if root != Path(config.DATA_DIR).resolve():
        return None  # a different profile's data never borrows this installation's counter

    def capture(changes):
        settings = _current_profile()
        return _root_counts(root, settings, changes) if settings is not None else None

    if retain is None and getattr(author, "task_id", None):
        retain = partial(_retain_proposal, author)
    return NanoPublication(root, lambda: _profile_witness(root), capture, retain, profile_key=_profile_key,
                           profile_stamp=profile_stamp)


def measure_publication_floor(root: Path) -> dict:
    """Count the same ordinary fresh-root envelope as a writer, without a proposal or model generation."""
    from ouroboros.local_model import local_dispatch_port
    from ouroboros.nano_publication import SOURCE_READ_INCOMPLETE, _formatter_digest, _invalidate_publication_reads, _publication_lock

    root = Path(root).resolve()
    admission = host_admission(root)
    if admission is None or not admission.revision():
        return {"status": "unknown", "reason": "active_profile_or_counter_unavailable"}
    pair = admission.capture(())
    if not pair:
        return {"status": "unknown", "reason": "no_complete_root_capture"}
    sample = pair[0].count()
    incomplete = sample.get("reason_code") == SOURCE_READ_INCOMPLETE
    if ((sample.get("status") == "exact" or incomplete)
            and (key := serving_counter_key({**sample, "port": local_dispatch_port()}))):
        bindings = {field: value for field, value in {"checked_formatter": _formatter_digest(sample),
                                                      **profile_stamp()}.items() if value}
        with _publication_lock(root) as marker:
            current = source_bytes(marker)
            if _checked_profile(current) == key and _marker_field(current, "reason") == "confirmed_counter_unavailable":
                _invalidate_publication_reads(marker, checked_profile=key, **bindings)
    return sample


def rules_save_ticket(root: Path, proposed: str):
    """A generic Settings save that changes the owner's rules counts them first, before the document lock.

    ``None`` where the save leaves the rules as they are. The count and its refusals are the
    dedicated writer's (``nano_activation.owner_rules_ticket``): both editors share one check.
    """
    from ouroboros.config import load_settings
    from ouroboros.nano_activation import owner_rules_ticket
    from ouroboros.safety_config import OWNER_RULES_KEY
    from ouroboros.settings_integrity import SettingsDocumentUnknown, skill_settings

    try:
        current = str(skill_settings(load_settings, strict=True).get(OWNER_RULES_KEY) or "")
    except SettingsDocumentUnknown as exc:
        raise NanoPublicationRefused("unavailable", "the current rules could not be read") from exc
    proposed = str(proposed or "")
    return None if proposed == current else owner_rules_ticket(root, proposed)


def settings_body_rules_ticket(body):
    """The count of a generic Settings save that authors the owner's rules, or ``None``.

    Only a Cyber save may author them there (``gateway.settings._merge_settings_payload``); every
    other mode ignores the key, and the commit's own check refuses rules that land uncounted.
    """
    from ouroboros import config
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import runtime_mode_at_least
    from ouroboros.safety_config import OWNER_RULES_KEY

    if not (isinstance(body, dict) and OWNER_RULES_KEY in body
            and runtime_mode_at_least(get_runtime_mode(), "cyber_pro")):
        return None
    return rules_save_ticket(config.DATA_DIR, str(body.get(OWNER_RULES_KEY) or ""))


def rules_commit_ticket(root: Path, before, after, ticket):
    """The ticket a Settings commit enters for the rules it writes, decided inside the document lock.

    A ticket counted for exactly these rules stays; one counted for other text never lands beside
    these bytes. Changed rules on the qualified profile without their count save nothing, so every
    editor of the rules (the dedicated endpoint, a Cyber save, an import) meets the same check.
    """
    from ouroboros import config
    from ouroboros.safety_config import OWNER_RULES_KEY

    old, new = (str((document or {}).get(OWNER_RULES_KEY) or "").encode("utf-8") for document in (before, after))
    counted = next((change for change in getattr(ticket, "changes", ()) if change.subject == "owner_rules"), None)
    if counted is not None and (counted.before, counted.after) == (old, new):
        return ticket
    if old == new:
        return None
    if counted is not None:
        raise NanoPublicationRefused("stale", "the rules changed while their capacity check ran; nothing was saved")
    if Path(root).resolve() != Path(config.DATA_DIR).resolve() or _current_profile() is None:
        return None
    raise NanoPublicationRefused("stale", "these rules were not checked against Nano's capacity; nothing was saved")
