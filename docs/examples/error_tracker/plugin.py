"""Send this process's ERROR log records to a Sentry-compatible error tracker (example).

The extension has no declared dependencies and no binary files, so it loads in
process in the server and in every pool worker. The SDK comes from Ouroboros's
own environment, and SENTRY_DSN names the destination; without it the SDK sends
nothing.
"""

import logging
import os

import sentry_sdk
from sentry_sdk.integrations.atexit import AtexitIntegration
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from ouroboros.observability import redact_projection


# The parts of an error event a tracker needs. The logging integration also copies a
# record's `extra` dict, which holds whatever a caller logged, so it is not forwarded.
_EVENT_FIELDS = ("event_id", "timestamp", "level", "logger", "platform", "sdk", "server_name", "release",
                 "environment", "tags", "fingerprint", "logentry", "message", "exception")


def _before_send(event, hint):
    return redact_projection({key: event[key] for key in _EVENT_FIELDS if key in event}).value


def register(api):
    role = "worker" if os.environ.get("OUROBOROS_IN_WORKER") == "1" else "server"
    sentry_sdk.init(
        include_local_variables=False,  # frame locals hold prompts, keys and file contents
        include_source_context=False,
        send_default_pii=False,
        default_integrations=False,  # no auto-instrumented model clients, no trace headers on their requests
        integrations=[
            LoggingIntegration(level=None, event_level=logging.ERROR),
            DedupeIntegration(),
            AtexitIntegration(),
        ],
        max_request_body_size="never",
        trace_propagation_targets=[],
        before_send=_before_send,
        server_name=f"ouroboros-{role}",  # not the machine's hostname
        release=f"ouroboros@{api.get_runtime_info().get('app_version', '')}",
    )
    api.on_unload(lambda: sentry_sdk.get_client().close(timeout=1))
