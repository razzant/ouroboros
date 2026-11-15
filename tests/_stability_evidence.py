"""Retain synthetic stability-server sources after physical teardown."""
import json
import shutil
import uuid

from ouroboros.test_evidence import diagnostic_directory


def retain_server_evidence(data, facts, output=b""):
    """Each generation keeps its own exact bytes, including model/tool evidence.

    Its facts and launcher output are published diagnostics; the data root's copy is a runtime
    archive and stays with the owner's run (``diagnostic_directory``'s default)."""
    from ouroboros import store
    # The external oracle must release its SQLite handles before the next server
    # generation changes store tiers, and before copying their settled bytes.
    store.close_connections_under(data)
    run = uuid.uuid4().hex
    shutil.copytree(data, diagnostic_directory(data.parent / ("retained-" + run), "stability-server-data") / "data",
                    dirs_exist_ok=True)
    kept = diagnostic_directory(data.parent / ("diagnostics-" + run), "stability-server", publish=True)
    (kept / "facts.json").write_text(json.dumps(facts, default=str))
    (kept / "launcher-output.bin").write_bytes(output)
