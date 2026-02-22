"""Multi-model review tool — sends code/text to multiple LLMs for consensus review.

Models are NOT hardcoded — the LLM chooses which models to use based on
prompt guidance. Budget is tracked via llm_usage events.
"""

import os
import json
import asyncio
import logging
import httpx

from ouroboros.utils import utc_now_iso
from ouroboros.tools.registry import ToolEntry, ToolContext


log = logging.getLogger(__name__)

# Maximum number of models allowed per review
MAX_MODELS = 10
# Concurrency limit for parallel requests
CONCURRENCY_LIMIT = 3  # Reduced to respect free tier rate limits

# Google AI Studio OpenAI-compatible endpoint
GOOGLE_AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def get_tools():
    """Return list of ToolEntry for registry."""
    return [
        ToolEntry(
            name="multi_model_review",
            schema={
                "name": "multi_model_review",
                "description": (
                    "Send code or text to multiple LLM models for review/consensus. "
                    "Each model reviews independently. Returns structured verdicts. "
                    "Choose diverse models yourself. Budget is tracked automatically."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The code or text to review",
                        },
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Review instructions — what to check for. "
                                "Fully specified by the LLM at call time."
                            ),
                        },
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Google AI Studio model IDs to query "
                                "(e.g. ['gemini-3.1-pro-preview', 'gemini-2.5-pro', 'gemini-2.0-flash'])"
                            ),
                        },
                    },
                    "required": ["content", "prompt", "models"],
                },
            },
            handler=_handle_multi_model_review,
        )
    ]


def _handle_multi_model_review(ctx: ToolContext, content: str = "", prompt: str = "", models: list = None) -> str:
    """Sync wrapper around async multi-model review. Registry calls this."""
    if models is None:
        models = []
    try:
        try:
            asyncio.get_running_loop()
            # Already in async context — run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, _multi_model_review_async(content, prompt, models, ctx)).result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run directly
            result = asyncio.run(_multi_model_review_async(content, prompt, models, ctx))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Multi-model review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


async def _query_model(client, model, messages, api_key, semaphore):
    """Query a single model with semaphore-based concurrency control. Returns (model, response_dict, headers_dict) or (model, error_str, None)."""
    async with semaphore:
        try:
            resp = await client.post(
                GOOGLE_AI_STUDIO_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                },
                timeout=120.0,
            )

            # Extract ALL data while client is still open
            status_code = resp.status_code
            response_text = resp.text
            response_headers = dict(resp.headers)

            if status_code != 200:
                error_text = response_text[:200]
                if len(response_text) > 200:
                    error_text += " [truncated]"
                return model, f"HTTP {status_code}: {error_text}", None

            data = resp.json()
            return model, data, response_headers
        except asyncio.TimeoutError:
            return model, "Error: Timeout after 120s", None
        except Exception as e:
            error_msg = str(e)[:200]
            if len(str(e)) > 200:
                error_msg += " [truncated]"
            return model, f"Error: {error_msg}", None


async def _multi_model_review_async(content: str, prompt: str, models: list, ctx: ToolContext):
    """Async orchestration: validate → query → parse → emit → return."""
    # Validation
    if not content:
        return {"error": "content is required"}
    if not prompt:
        return {"error": "prompt is required"}
    if not models:
        return {"error": "models list is required (e.g. ['gemini-3.1-pro-preview', 'gemini-2.5-pro', 'gemini-2.0-flash'])"}

    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        return {"error": "models must be a list of strings"}

    if len(models) > MAX_MODELS:
        return {"error": f"Too many models requested ({len(models)}). Maximum is {MAX_MODELS}."}

    if len(models) == 0:
        return {"error": "At least one model is required"}

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return {"error": "GOOGLE_API_KEY not set"}

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]

    # Query all models with bounded concurrency
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with httpx.AsyncClient() as client:
        tasks = [_query_model(client, m, messages, api_key, semaphore) for m in models]
        results = await asyncio.gather(*tasks)

    # Parse and process results
    review_results = []
    for model, result, headers_dict in results:
        review_result = _parse_model_response(model, result, headers_dict)
        _emit_usage_event(review_result, ctx)
        review_results.append(review_result)

    return {
        "model_count": len(models),
        "results": review_results,
    }


def _parse_model_response(model: str, result, headers_dict) -> dict:
    """Parse one model's response into structured review_result dict."""
    if isinstance(result, str):
        # Error case
        return {
            "model": model,
            "verdict": "ERROR",
            "text": result,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_estimate": 0.0,
        }

    # Success case — extract response text and verdict
    try:
        choices = result.get("choices", [])
        if not choices:
            text = f"(no choices in response: {json.dumps(result)[:200]})"
            verdict = "ERROR"
        else:
            text = choices[0]["message"]["content"]
            # Robust verdict parsing: check first 3 lines for PASS/FAIL anywhere (case-insensitive)
            verdict = "UNKNOWN"
            lines = text.split("\n")[:3]  # Check only first 3 lines
            for line in lines:
                line_upper = line.upper()
                if "PASS" in line_upper:
                    verdict = "PASS"
                    break
                elif "FAIL" in line_upper:
                    verdict = "FAIL"
                    break
    except (KeyError, IndexError, TypeError):
        error_text = json.dumps(result)[:200]
        if len(json.dumps(result)) > 200:
            error_text += " [truncated]"
        text = f"(unexpected response format: {error_text})"
        verdict = "ERROR"

    # Extract usage for budget tracking
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Extract cost from response body (preferred) or headers
    cost = 0.0
    try:
        # First check response body for usage.cost
        if "usage" in result and "cost" in result["usage"]:
            cost = float(result["usage"]["cost"])
        # Fallback to total_cost field
        elif "usage" in result and "total_cost" in result["usage"]:
            cost = float(result["usage"]["total_cost"])
        # Finally check headers
        elif headers_dict:
            # Case-insensitive search for cost header
            for key, value in headers_dict.items():
                if key.lower() == "x-openrouter-cost":
                    cost = float(value)
                    break
    except (ValueError, TypeError, KeyError):
        pass

    return {
        "model": model,
        "verdict": verdict,
        "text": text,
        "tokens_in": prompt_tokens,
        "tokens_out": completion_tokens,
        "cost_estimate": cost,
    }


def _emit_usage_event(review_result: dict, ctx: ToolContext) -> None:
    """Emit llm_usage event for budget tracking (for ALL cases, including errors)."""
    if ctx is None:
        return

    usage_event = {
        "type": "llm_usage",
        "ts": utc_now_iso(),
        "task_id": ctx.task_id if ctx.task_id else "",
        "usage": {
            "prompt_tokens": review_result["tokens_in"],
            "completion_tokens": review_result["tokens_out"],
            "cost": review_result["cost_estimate"],
        },
        "category": "review",
    }

    if ctx.event_queue is not None:
        try:
            ctx.event_queue.put_nowait(usage_event)
        except Exception:
            # Fallback to pending_events if queue fails
            if hasattr(ctx, "pending_events"):
                ctx.pending_events.append(usage_event)
    elif hasattr(ctx, "pending_events"):
        # No event_queue — use pending_events
        ctx.pending_events.append(usage_event)
