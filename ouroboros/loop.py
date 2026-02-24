                # If fallback also fails, enter persistent retry loop
                # We don't give up — network issues may be transient. Keep trying until success or task killed.
                retry_interval = int(os.environ.get("OUROBOROS_FALLBACK_RETRY_INTERVAL", "10"))  # seconds
                emit_progress(f"⚠️ Fallback model {fallback_model} also failed. Entering retry loop (every {retry_interval}s) until network recovers.")
                while msg is None:
                    time.sleep(retry_interval)
                    emit_progress(f"🔄 Retrying fallback model {fallback_model}...")
                    msg, fallback_cost = _call_llm_with_retry(
                        llm, messages, fallback_model, tool_schemas, active_effort,
                        max_retries, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type
                    )
                    if msg is not None:
                        break
                # Fallback succeeded — continue processing with this msg