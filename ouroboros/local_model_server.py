"""The owned llama-cpp-python server: an input-measurement route, and a replayed tool call rendered with object
arguments when a model's chat template fails on the JSON text the OpenAI wire carries (Qwen3.8's does)."""

import hashlib
import inspect
import json
import os
from typing import Any


def input_fingerprint(payload: dict) -> str:
    """The native formatter's complete input, independent of generation length."""
    keys = ("model", "messages", "functions", "function_call", "tools", "tool_choice")
    value = {key: payload.get(key) for key in keys}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()



def object_tool_arguments(messages: Any) -> Any:
    """Messages whose replayed tool-call arguments are objects instead of the JSON text the OpenAI wire carries.

    Hugging Face chat templates may iterate a call's arguments (``tool_call.arguments|items`` in Qwen3.5 and later);
    on text that fails, so the second round of every tool task would. Text that is not a JSON object stays."""
    if not isinstance(messages, list):
        return messages
    converted = []
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if isinstance(calls, list):
            fixed = []
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                arguments = function.get("arguments") if isinstance(function, dict) else None
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        call = {**call, "function": {**function, "arguments": parsed}}
                fixed.append(call)
            message = {**message, "tool_calls": fixed}
        converted.append(message)
    return converted


def render_tool_arguments_on_retry(render):
    """A formatter call that renders messages as given and, only if the template fails on them, again with object
    arguments: a template that prints the text keeps it, one that needs a mapping gets one (measurement too)."""
    def call(self, *, messages, **kwargs):
        try:
            return render(self, messages=messages, **kwargs)
        except Exception:
            objects = object_tool_arguments(messages)
            if objects == messages:
                raise
            return render(self, messages=objects, **kwargs)

    call._ouroboros_object_arguments = True
    return call


def install_object_tool_arguments() -> None:
    """Patch this server process's Jinja formatter once (the vendor server renders every model template with it)."""
    from llama_cpp import llama_chat_format

    formatter = llama_chat_format.Jinja2ChatFormatter
    if not getattr(formatter.__call__, "_ouroboros_object_arguments", False):
        formatter.__call__ = render_tool_arguments_on_retry(formatter.__call__)

def measure_chat_input(model: Any, payload: dict) -> dict:
    """Render through the selected provider formatter without running inference.

    Only the provider's generic single-completion adapter establishes this
    proof. Custom/multimodal/multiple-generation handlers continue to serve
    ordinary requests but do not acquire a false count or output guarantee.
    """
    import llama_cpp
    from llama_cpp import llama_chat_format

    result = {"supported": False, "input_is_exact": False, "input_tokens": None,
              "context_window": int(model.n_ctx()), "process_id": os.getpid(),
              "native_input_sha256": input_fingerprint(payload),
              "output_limit_enforced": False, "reasoning_included_in_limit": None,
              "reason": "selected_formatter_not_measurable"}
    handler = (model.chat_handler or model._chat_handlers.get(model.chat_format)
               or llama_chat_format.get_chat_completion_handler(model.chat_format))
    generic = llama_chat_format.chat_formatter_to_chat_completion_handler(lambda **kwargs: None)
    if getattr(handler, "__code__", None) is not generic.__code__:
        return result
    formatter = inspect.getclosurevars(handler).nonlocals.get("chat_formatter")
    if not callable(formatter):
        return result
    try:
        formatted = formatter(**{key: payload.get(key) for key in
            ("messages", "functions", "function_call", "tools", "tool_choice")})
        tokens = model.tokenize(formatted.prompt.encode("utf-8"),
                                add_bos=not formatted.added_special, special=True)
    except Exception as error:
        return {**result, "reason": f"formatter_measurement_failed:{type(error).__name__}"}
    return {**result, "supported": True, "input_is_exact": True, "input_tokens": len(tokens),
            "output_limit_enforced": True, "reasoning_included_in_limit": True, "reason": None,
            "tokenizer_template_provenance": {
                "source": "serving_llama_formatter", "library_version": llama_cpp.__version__,
                "chat_format": str(model.chat_format or ""),
                "formatter": type(formatter).__name__,
                "rendered_prompt_sha256": hashlib.sha256(formatted.prompt.encode("utf-8")).hexdigest(),
            }}


def with_measurement_route(app):
    """Reuse the server's authentication, model proxy and inference exclusion lock."""
    from fastapi import Depends
    from llama_cpp.server import app as server_app
    from llama_cpp.server.types import CreateChatCompletionRequest

    async def measurement_model():
        # The outer lock tells a live stream to stop. This read must only wait
        # on the existing inference lock, never signal cancellation to a peer.
        async with server_app.llama_inner_lock:
            yield server_app._llama_proxy

    @app.post("/extras/measure_chat", dependencies=[Depends(server_app.authenticate)])
    async def measure(body: CreateChatCompletionRequest, proxy=Depends(measurement_model)):
        return measure_chat_input(proxy(body.model), body.model_dump())

    return app


def main():
    """Use the vendor's CLI/configuration/uvicorn lifecycle without a second server."""
    from llama_cpp.server import __main__ as server

    install_object_tool_arguments()
    create_app = server.create_app
    server.create_app = lambda *args, **kwargs: with_measurement_route(create_app(*args, **kwargs))
    try:
        server.main()
    finally:
        server.create_app = create_app


if __name__ == "__main__":
    main()
