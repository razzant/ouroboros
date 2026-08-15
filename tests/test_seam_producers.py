"""CLASS GATE 1 — a declared seam must have a PRODUCTION producer, or say it has none.

The defect this closes appeared at least a dozen times in the remote-workspaces feature:
a method, channel or contract half was written, documented, and covered by a test, and
NOTHING in production ever called it.  ``execute_prepared`` was the expensive instance —
the whole execution phase was dead and the suite was green, because the tests called the
seam directly.  Others: ``cancel_admission``, ``abort_prepared``,
``builtin_execution_affinity``, a browser-forward pass-through, five export channels
declared without a Home half, and the task→session binding.

Listing those is not a fix.  This gate makes the CLASS checkable: for an ENUMERATED set
of seams, every public member must resolve to at least one production reference, or be
named in ``SEAM_WITHOUT_PRODUCER`` with a verdict and a reason.  Three verdicts are
allowed, and they are the only three honest answers to "nobody calls this":

* ``test_facing`` — it exists so a test can assert a contract. Legitimate, but it must be
  SAID, because a test-only accessor proves nothing about the shipped path.
* ``deferred``   — the other half of the contract is genuinely not built yet. Must name
  the consequence, so the gap is a decision on the record rather than an oversight.
* ``protocol``   — a Protocol/ABC member implemented elsewhere and called through the
  interface.

BOUNDARY of this gate, stated because an unstated boundary is how the previous version
of this idea failed (see ``tests/test_source_gate_boundaries.py``, which asserts that
this docstring keeps saying so):

* Producers are resolved by NAME plus RECEIVER CAPABILITY: a reference counts only if it
  appears in a module that can actually obtain an instance of the owning class (the
  defining module, or a module naming the class / its protocol / its factory).  Plain
  name matching alone is NOT enough and was measured to be wrong here: with it,
  ``SpoolStreamSink.acknowledge`` and ``.expire`` looked alive because unrelated classes
  have methods of the same name with real callers.
* The resolver counts attribute access, ``getattr(x, "name")``, keyword strings, and any
  string literal containing the name as a token — so dynamic dispatch is a producer.  It
  therefore OVER-approximates: an unrelated string that happens to contain the name will
  mask an orphan.  That direction is chosen deliberately (a false green on one member is
  cheaper than a gate people delete), and the registry is the compensating control.
* The seam set is ENUMERATED, not inferred.  A new seam class is invisible to this gate
  until someone adds it, which is why ``test_the_seam_set_still_covers_the_feature``
  asserts the enumeration against the feature's own module list.
"""

from __future__ import annotations

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

# class name -> (defining module, extra identifiers through which a caller can obtain one)
# The handles are deliberately SPECIFIC names (class, protocol, factory). A generic word
# like "ledger" or "transport" makes almost every module "capable" and the gate blind.
SEAM_CLASSES: dict[str, tuple[str, tuple[str, ...]]] = {
    "RemoteSessionBroker": (
        "ouroboros/remote_workspace.py",
        ("RemoteSessionBroker", "get_remote_workspace_service"),
    ),
    "RemoteWorkspacePipeProxy": (
        "ouroboros/remote_worker_proxy.py",
        ("RemoteWorkspacePipeProxy",),
    ),
    # `RemoteTransport` is the Protocol the broker holds one of; a module that names it
    # (including in the comment that explains why its own annotation is `Any`) is a
    # module that can call through it.
    "OpenSSHExecdTransport": (
        "ouroboros/remote_ssh.py",
        ("OpenSSHExecdTransport", "RemoteTransport"),
    ),
    "RemoteTransferService": ("ouroboros/remote_transfer.py", ("RemoteTransferService",)),
    "ExecdService": ("ouroboros/execd.py", ("ExecdService",)),
    "ExecdNativeControl": ("ouroboros/execd.py", ("ExecdNativeControl", "NativeExecutionControl")),
    "CASBlobStore": ("ouroboros/execd_state.py", ("CASBlobStore",)),
    "OperationJournal": ("ouroboros/execd_state.py", ("OperationJournal",)),
    "LeaseCustody": ("ouroboros/execd_state.py", ("LeaseCustody",)),
    "SSHBrowserForwardManager": (
        "ouroboros/remote_browser_forward.py",
        ("SSHBrowserForwardManager",),
    ),
    "RemoteServiceLeaseBook": (
        "ouroboros/remote_service_leases.py",
        ("RemoteServiceLeaseBook",),
    ),
    "RemoteWorkspaceSnapshot": (
        "ouroboros/remote_transfer.py",
        ("RemoteWorkspaceSnapshot", "materialize_remote_snapshot"),
    ),
    "WorkerChannels": ("ouroboros/remote_worker_proxy.py", ("WorkerChannels",)),
    "RemoteTaskFileCache": ("ouroboros/execd_task_files.py", ("RemoteTaskFileCache",)),
    "ProcessLogSpool": (
        "ouroboros/execd_spool.py",
        ("ProcessLogSpool", "ProcessSpool"),
    ),
    "SpoolStreamSink": (
        "ouroboros/execd_spool.py",
        ("SpoolStreamSink", "ProcessSpoolSink", "open_process_spool_sinks"),
    ),
    "SpoolQuotaLedger": ("ouroboros/execd_spool.py", ("SpoolQuotaLedger",)),
    "PendingJournal": ("ouroboros/remote_transfer.py", ("PendingJournal",)),
    "HomeImporter": ("ouroboros/remote_transfer.py", ("HomeImporter",)),
}

# (class, member) -> (verdict, reason). Verdicts: test_facing | deferred | protocol.
SEAM_WITHOUT_PRODUCER: dict[tuple[str, str], tuple[str, str]] = {
    ("ProcessLogSpool", "read_sealed"): (
        "deferred",
        "D8 sealed process-log EXPORT still has no Home half: no operation fetches a "
        "sealed blob, so a spooled log is written and never read (service_logs answers "
        "from the LIVE spool). Materialization on the model's demand is deferred by "
        "owner decision — it is a protocol addition. What is NO LONGER deferred is the "
        "consequence: the quota those blobs hold is released by "
        "ProcessLogSpool.release_task at the task terminal and by expire_retained on the "
        "custody tick, so an unfetched log costs disk until retention, not forever.",
    ),
    ("SpoolStreamSink", "acknowledge"): (
        "deferred",
        "the Home-side ACK that would settle one stream is still unwired: "
        "execd.acknowledge() answers the OPERATION journal, not the stream sink. It is a "
        "finer grain than the release that now exists — a sink is a per-operation object "
        "that dies with the operation, so the durable unit is the task, and "
        "ProcessLogSpool.release_task frees the quota and the blobs from the retention "
        "index instead. This member would only add per-stream precision.",
    ),
    ("SSHBrowserForwardManager", "records"): (
        "test_facing",
        "the live-forward view three cleanup cases in tests/test_remote_browser_forward.py "
        "assert emptiness against. Production never enumerates forwards — panic and task "
        "cleanup act on the internal map directly — so this proves the map is empty, not "
        "that any shipped path reads it.",
    ),
    ("OperationJournal", "list_records"): (
        "test_facing",
        "the journal's read-side view, asserted by nine cases in tests/test_execd_state.py "
        "and called by no production path: execd answers operations from the live record, "
        "never by listing. Kept because the tests that use it are real coverage of the "
        "journal's own accounting, but it proves nothing about a shipped route.",
    ),
    ("SpoolQuotaLedger", "usage"): (
        "test_facing",
        "a read-only view of the quota ledger that exists so tests can assert host and "
        "per-task accounting after reserve/release. No production path reports quota "
        "usage anywhere, which is a real observability gap but not a broken contract.",
    ),
    ("SpoolStreamSink", "expire"): (
        "deferred",
        "the retention sweeper does not go through the SINK, and cannot: by the time a "
        "blob ages out the operation that owned the sink is long gone. "
        "ProcessLogSpool.expire_retained works from the durable retention index instead, "
        "so RetentionExpired is still only ever dispatched from inside seal() for "
        "payloads small enough to inline. What this member was standing in for — a host "
        "accumulating sealed blobs until the quota refuses every new reservation — is "
        "closed; this entry point is not the one that closed it.",
    ),
}


def _production_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for pattern in ("ouroboros/**/*.py", "supervisor/**/*.py"):
        out.extend(p for p in REPO.glob(pattern) if "__pycache__" not in str(p))
    for extra in ("server.py", "launcher.py"):
        if (REPO / extra).exists():
            out.append(REPO / extra)
    return sorted(out)


def _load():
    src: dict[str, str] = {}
    trees: dict[str, ast.Module] = {}
    for path in _production_files():
        rel = str(path.relative_to(REPO))
        try:
            text = path.read_text(encoding="utf-8")
            trees[rel] = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        src[rel] = text
    return src, trees


def _receiver_capable(src: dict[str, str], rel: str, handles: tuple[str, ...]) -> list[str]:
    """Modules that could hold an instance: the definer, or one naming a handle."""

    return [
        module
        for module, text in src.items()
        if module == rel
        or any(re.search(rf"\b{re.escape(h)}\b", text) for h in handles)
    ]


def _producers(
    trees,
    modules: list[str],
    member: str,
    def_site: tuple[str, int],
    own_span: tuple[int, int] = (0, 0),
) -> list[str]:
    """Production references to ``member``, excluding the member's own body.

    Excluding the body matters: the string matcher tokenises every literal, so a member
    whose docstring merely says the word ("Live forward records.") would otherwise count
    as its own producer and the gate would report itself green. A member cannot produce
    itself.
    """

    hits: list[str] = []
    for module in modules:
        for node in ast.walk(trees[module]):
            in_own_body = (
                module == def_site[0]
                and own_span[0] <= getattr(node, "lineno", 0) <= own_span[1]
            )
            if in_own_body:
                continue
            if isinstance(node, ast.Attribute) and node.attr == member:
                if (module, node.lineno) != def_site:
                    hits.append(f"{module}:{node.lineno}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if member in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", node.value):
                    hits.append(f"{module}:{node.lineno} (string dispatch)")
    return hits


def test_every_enumerated_seam_member_has_a_production_producer():
    """A public seam member is called in production, or declared unproduced with a reason."""

    src, trees = _load()
    violations: list[str] = []
    for cls, (rel, handles) in SEAM_CLASSES.items():
        assert rel in trees, f"seam module {rel} vanished; update SEAM_CLASSES"
        node = next(
            (n for n in ast.walk(trees[rel]) if isinstance(n, ast.ClassDef) and n.name == cls),
            None,
        )
        assert node is not None, f"seam class {cls} not found in {rel}; update SEAM_CLASSES"
        capable = _receiver_capable(src, rel, handles)
        for sub in node.body:
            if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if sub.name.startswith("_"):
                continue
            span = (sub.lineno, sub.end_lineno or sub.lineno)
            if _producers(trees, capable, sub.name, (rel, sub.lineno), span):
                continue
            if (cls, sub.name) in SEAM_WITHOUT_PRODUCER:
                continue
            violations.append(
                f"{rel}:{sub.lineno}: {cls}.{sub.name} has NO production producer.\n"
                f"        Lay a call, delete the member, or declare it in "
                f"SEAM_WITHOUT_PRODUCER with a verdict and a reason."
            )
    assert not violations, (
        "Declared seam with no producer — the class that shipped a dead execution "
        "phase behind a green suite:\n" + "\n".join(violations)
    )


def test_declared_unproduced_seams_are_still_unproduced_and_still_exist():
    """The registry cannot rot in either direction.

    A member that GAINED a producer must leave the registry (otherwise the exemption
    silently covers a live seam and a later regression goes unseen), and a member that
    was deleted must leave too (otherwise the registry describes code nobody has).
    """

    src, trees = _load()
    stale: list[str] = []
    for (cls, member), (verdict, reason) in SEAM_WITHOUT_PRODUCER.items():
        assert verdict in {"test_facing", "deferred", "protocol"}, (
            f"{cls}.{member}: unknown verdict {verdict!r}"
        )
        assert len(reason) > 60, (
            f"{cls}.{member}: a deferral needs its CONSEQUENCE in words, not a label"
        )
        rel, handles = SEAM_CLASSES[cls]
        node = next(
            (n for n in ast.walk(trees[rel]) if isinstance(n, ast.ClassDef) and n.name == cls),
            None,
        )
        assert node is not None, f"{cls} is gone; drop its rows from SEAM_WITHOUT_PRODUCER"
        member_node = next(
            (
                s
                for s in node.body
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == member
            ),
            None,
        )
        if member_node is None:
            stale.append(f"{cls}.{member} no longer exists — remove the registry row")
            continue
        capable = _receiver_capable(src, rel, handles)
        found = _producers(
            trees,
            capable,
            member,
            (rel, member_node.lineno),
            (member_node.lineno, member_node.end_lineno or member_node.lineno),
        )
        if found:
            stale.append(
                f"{cls}.{member} now HAS a producer ({found[0]}) — remove the registry row "
                "so the seam is gated like every other live one"
            )
    assert not stale, "SEAM_WITHOUT_PRODUCER is stale:\n" + "\n".join(stale)


def test_every_declared_export_channel_producer_resolves_to_a_called_symbol():
    """A channel's producer must be a real symbol with a real caller, not a string.

    The existing registry gate (tests/test_remote_export_policy.py) asserts each channel
    names a producer and that the string is non-empty — which a typo, a rename or a
    deleted function all survive. This resolves the string.
    """

    from ouroboros.remote_export_policy import HOME_CHANNEL_PRODUCERS

    _src, trees = _load()
    broken: list[str] = []
    for channel, ref in HOME_CHANNEL_PRODUCERS.items():
        module_part, _, symbol = ref.rpartition(".")
        rel = f"ouroboros/{module_part}.py"
        if rel not in trees:
            broken.append(f"{channel}: producer names module {rel}, which does not exist")
            continue
        defined = [
            n
            for n in ast.walk(trees[rel])
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol
        ]
        if not defined:
            broken.append(f"{channel}: {rel} defines no {symbol!r}")
            continue
        def_lines = {n.lineno for n in defined}
        callers = [
            f"{module}:{node.lineno}"
            for module, tree in trees.items()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                getattr(node.func, "attr", None) == symbol
                or getattr(node.func, "id", None) == symbol
            )
            and not (module == rel and node.lineno in def_lines)
        ]
        if not callers:
            broken.append(
                f"{channel}: {ref} exists but nothing in production calls it — the "
                "channel is declared with a producer that never produces"
            )
    assert not broken, (
        "Export channel producers that do not resolve:\n" + "\n".join(broken)
    )


def test_the_seam_set_still_covers_the_feature():
    """A new remote seam class must be enumerated, or this gate quietly shrinks.

    Checks the weaker but maintainable property: every feature module that defines a
    class with 4+ public methods is represented in SEAM_CLASSES. That threshold is what
    separates a service boundary from a value object, and it is stated rather than
    tuned — raising it to hide a new class is a visible edit.
    """

    feature = sorted(
        set(REPO.glob("ouroboros/remote_*.py")) | set(REPO.glob("ouroboros/execd*.py"))
    )
    missing: list[str] = []
    for path in feature:
        rel = str(path.relative_to(REPO))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            public = [
                s
                for s in node.body
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not s.name.startswith("_")
            ]
            if len(public) >= 4 and node.name not in SEAM_CLASSES:
                missing.append(f"{rel}:{node.lineno}: {node.name} ({len(public)} public members)")
    assert not missing, (
        "Feature classes wide enough to be a seam but absent from SEAM_CLASSES — add "
        "them (with their receiver handles) or this gate does not police them:\n"
        + "\n".join(missing)
    )
