                # If fallback also fails, enter persistent retry loop
                # We don't give up — network issues may be transient. Keep trying until success or task killed.
                retry_interval = int(os.environ.get("OUROBOROS_FALLBACK_RETRY_INTERVAL", "10"))  # seconds
                emit_progress(f"⚠️ Fallback model {fallback_model} also failed. Entering retry loop (every {retry_interval}s) until network recovers.")

                # Determine retry limit: unlimited in production, limited in tests to avoid hang
                if os.environ.get('PYTEST_CURRENT_TEST'):
                    max_attempts = int(os.environ.get("OUROBOROS_FALLBACK_TEST_MAX_ATTEMPTS", "3"))
                else:
                    max_attempts = None  # unlimited

                attempts = 0
                while msg is None:
                    if max_attempts is not None:
                        attempts += 1
                        if attempts > max_attempts:
                            emit_progress(f"❌ Fallback retry limit ({max_attempts}) reached. Giving up.")
                            break
                    time.sleep(retry_interval)
                    emit_progress(f"🔄 Retrying fallback model {fallback_model}... (attempt {attempts})" if max_attempts else f"🔄 Retrying fallback model {fallback_model}...")
                    msg, fallback_cost = _call_llm_with_retry(
                        llm, messages, fallback_model, tool_schemas, active_effort,
                        max_retries, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type
                    )

                # If still no response after retries, return error
                if msg is None:
                    return (
                        f"⚠️ Failed to get a response from model {fallback_model} after fallback and {attempts} retries. Network issues persist. "
                        f"Consider rephrasing your request or waiting for connectivity to recover."
                    ), accumulated_usage, llm_trace

                # Fallback succeeded — continue processing with this msg