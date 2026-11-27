"""The managed launch of the owned llama-cpp server keeps last-position logits unless its environment chooses.

Keyless and model-free: ``Popen`` records instead of starting a process, and the vendor modules ``main`` imports are
stand-ins, never the vendor package.
"""
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ouroboros import local_model, local_model_server
from ouroboros.local_model_server import with_last_position_logits

_LOGITS_SETTINGS = ("logits_all", "draft_model", "config_file")


@pytest.mark.parametrize("environ, added", [
    ({}, True),
    ({"PATH": "/usr/bin"}, True),
    ({"LOGITS_ALL": "true"}, False),
    ({"logits_all": "1"}, False),  # the vendor reads setting names in any letter case
    ({"LOGITS_ALL": ""}, False),  # an explicit value, malformed or not, is the vendor's to judge
    ({"draft_model": "prompt-lookup-decoding"}, False),
    ({"CONFIG_FILE": "/etc/llama.json"}, False),
    ({"config_file": "/etc/llama.json"}, True),  # the vendor looks the config file up by its exact name only
])
def test_only_an_environment_that_configures_no_logits_choice_gets_the_default(environ, added):
    given = dict(environ)
    result = with_last_position_logits(environ)
    assert environ == given and result is not environ
    assert result == ({**given, "LOGITS_ALL": "false"} if added else given)


def _clear_logits_settings(monkeypatch):
    for key in [key for key in os.environ if key.lower() in _LOGITS_SETTINGS]:
        monkeypatch.delenv(key)


def _managed_launch(monkeypatch):
    """``start_server`` with a recording ``Popen``: the argv and child environment it would start the server with."""
    launched = {}

    def popen(cmd, **kwargs):  # no process starts; the stand-in has no pid, so nothing can signal one
        launched.update(cmd=list(cmd), env=dict(kwargs["env"]))
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(local_model, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": popen}))
    monkeypatch.setattr(local_model, "_run_runtime_probe", lambda: SimpleNamespace(returncode=0))
    manager = local_model.LocalModelManager()
    for name in ("_drain_stderr", "_wait_for_healthy"):
        monkeypatch.setattr(manager, name, lambda: None)
    process_environment = dict(os.environ)
    try:
        manager.start_server("/models/fixture.gguf", port=8123, n_gpu_layers=0, n_ctx=80000, chat_format="chatml")
    finally:
        manager._proc = None
    assert dict(os.environ) == process_environment  # only the child's copy carries the default
    return launched["cmd"], launched["env"]


@pytest.mark.parametrize("operator, child", [
    ({}, {"logits_all": "false"}),
    ({"LOGITS_ALL": "true"}, {"logits_all": "true"}),
    ({"LOGITS_ALL": "false"}, {"logits_all": "false"}),
    ({"DRAFT_MODEL": "prompt-lookup-decoding"}, {"draft_model": "prompt-lookup-decoding"}),
    ({"CONFIG_FILE": "/etc/llama.json"}, {"config_file": "/etc/llama.json"}),
])
def test_the_managed_launch_defaults_only_an_unconfigured_child_environment(monkeypatch, operator, child):
    _clear_logits_settings(monkeypatch)
    for key, value in operator.items():
        monkeypatch.setenv(key, value)
    cmd, env = _managed_launch(monkeypatch)
    assert {key.lower(): value for key, value in env.items() if key.lower() in _LOGITS_SETTINGS} == child
    # No command-line choice outranks the environment, and the window stays the pair the published binding reads.
    assert not [argument for argument in cmd if argument.startswith(("--logits_all", "--draft_model", "--config"))]
    line = " ".join(cmd)
    assert local_model._SERVING_CONTEXT_ARGV_RE.search(line).group(1) == "80000"
    assert local_model._SERVING_MODEL_ARGV_RE.search(line).groups() == ("/models/fixture.gguf", "8123")


def test_the_server_entry_leaves_a_direct_command_line_its_own_choices(monkeypatch):
    """``main`` adds no environment: a direct ``--draft_model`` or ``--config_file`` stays the vendor's to resolve."""
    read = []
    package, server_package = ModuleType("llama_cpp"), ModuleType("llama_cpp.server")
    chat_format, vendor_main = ModuleType("llama_cpp.llama_chat_format"), ModuleType("llama_cpp.server.__main__")
    chat_format.Jinja2ChatFormatter = type("Jinja2ChatFormatter", (), {"__call__": lambda self, **kwargs: None})
    vendor_main.create_app = lambda *args, **kwargs: None
    vendor_main.main = lambda: read.append(
        {key: value for key, value in os.environ.items() if key.lower() in _LOGITS_SETTINGS})
    package.llama_chat_format, package.server, server_package.__main__ = chat_format, server_package, vendor_main
    for module in (package, chat_format, server_package, vendor_main):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    _clear_logits_settings(monkeypatch)
    process_environment = dict(os.environ)
    local_model_server.main()
    assert read == [{}] and dict(os.environ) == process_environment
