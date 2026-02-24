...
                # Try fallback model (don't increment round_idx — this is still same logical round)
                msg, fallback_cost = _call_llm_with_retry(
                    llm, messages, fallback_model, tool_schemas, active_effort,
                    max_retries, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type
                )

                # If fallback also fails, give up
                if msg is None:
                    return (
                        f"⚠️ Failed to get a response from the model after {max_retries} attempts. "
                        f"Fallback model ({fallback_model}) also returned no response."
                    ), accumulated_usage, llm_trace
...