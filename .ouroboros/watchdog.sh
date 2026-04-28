#!/bin/bash
# Ouroboros Zera — Supervisor Watchdog
# Мониторит процесс supervisor и перезапускает его при падении.
# Также проверяет здоровье llama-server и Telegram API.
#
# Usage:
#   cd /home/zera/ouroboros_zera && bash .ouroboros/watchdog.sh
#
# Environment variables (optional overrides):
#   WATCHDOG_CHECK_INTERVAL_SEC   — interval between health checks (default: 30)
#   WATCHDOG_LLM_TIMEOUT_SEC      — max time for llama-server health check (default: 10)
#   WATCHDOG_TG_TIMEOUT_SEC       — max time for Telegram API check (default: 10)
#   WATCHDOG_LOG_FILE             — path to watchdog log (default: .ouroboros/logs/watchdog.jsonl)
#   WATCHDOG_MAX_RESTARTS_24H     — max restarts in 24h window (default: 20)
#   WATCHDOG_GRACE_PERIOD_SEC     — wait before restart after crash (default: 5)

set -u  # error on undefined vars, but not -e to avoid exiting on health check failures

# ----------------------------
# Configuration
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

CHECK_INTERVAL="${WATCHDOG_CHECK_INTERVAL_SEC:-30}"
LLM_TIMEOUT="${WATCHDOG_LLM_TIMEOUT_SEC:-10}"
TG_TIMEOUT="${WATCHDOG_TG_TIMEOUT_SEC:-10}"
LOG_FILE="${WATCHDOG_LOG_FILE:-$LOG_DIR/watchdog.jsonl}"
MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS_24H:-20}"
GRACE_PERIOD="${WATCHDOG_GRACE_PERIOD_SEC:-5}"

RUN_SH="$SCRIPT_DIR/run.sh"
PID_FILE="$SCRIPT_DIR/supervisor.pid"

# Load .env if exists (for TELEGRAM_BOT_TOKEN etc)
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs) 2>/dev/null || true
fi

# ----------------------------
# Logging
# ----------------------------
log_json() {
    local type="$1"
    shift
    local msg="$@"
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
    echo "{\"ts\": \"$ts\", \"type\": \"$type\", $msg}" >> "$LOG_FILE"
}

log_msg() {
    local type="$1"
    shift
    local msg="$@"
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
    echo "[$ts] [$type] $msg"
}

# ----------------------------
# Health checks
# ----------------------------
check_llama_health() {
    # Check if llama-server responds to /health endpoint
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout "$LLM_TIMEOUT" \
        --max-time "$LLM_TIMEOUT" \
        http://localhost:8080/health 2>/dev/null) || true
    
    if [ "$http_code" = "200" ]; then
        return 0
    else
        return 1
    fi
}

check_telegram_health() {
    # Check if Telegram API is reachable via the bot
    local response
    response=$(curl -s -m "$TG_TIMEOUT" \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN:-}/getMe" 2>/dev/null) || true
    
    if echo "$response" | grep -q '"ok":true' 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# ----------------------------
# Supervisor process management
# ----------------------------
get_supervisor_pid() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    
    # Fallback: find by command line
    local pid
    pid=$(pgrep -f "python.*local_launcher.py" -o 2>/dev/null) || true
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "$pid"
        return 0
    fi
    
    return 1
}

is_supervisor_alive() {
    get_supervisor_pid >/dev/null 2>&1
}

restart_supervisor() {
    log_msg "WATCHDOG" "Restarting supervisor..."
    log_json "watchdog_restart" "\"pid\": null, \"reason\": \"$1\""
    
    # Kill old process if still running
    local old_pid
    old_pid=$(get_supervisor_pid 2>/dev/null) || true
    if [ -n "$old_pid" ]; then
        log_msg "WATCHDOG" "Killing old supervisor PID $old_pid"
        kill -TERM "$old_pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$old_pid" 2>/dev/null || true
    fi
    
    # Wait grace period
    log_msg "WATCHDOG" "Waiting ${GRACE_PERIOD}s before restart..."
    sleep "$GRACE_PERIOD"
    
    # Start supervisor
    cd "$PROJECT_DIR" || exit 1
    nohup bash "$RUN_SH" >> "$LOG_DIR/supervisor-output.log" 2>&1 &
    local new_pid=$!
    
    # Save PID
    echo "$new_pid" > "$PID_FILE"
    
    log_msg "WATCHDOG" "Supervisor restarted with PID $new_pid"
    log_json "watchdog_restarted" "\"pid\": $new_pid"
}

# ----------------------------
# Restart rate limiting
# ----------------------------
check_restart_rate() {
    local restart_log="$LOG_DIR/restart_history.log"
    local now
    now=$(date +%s)
    local window_start=$((now - 86400))  # 24 hours
    
    # Clean old entries
    if [ -f "$restart_log" ]; then
        local temp_log="${restart_log}.tmp"
        awk -v start="$window_start" '{if ($1 >= start) print}' "$restart_log" > "$temp_log"
        mv "$temp_log" "$restart_log"
    fi
    
    # Count recent restarts
    local count=0
    if [ -f "$restart_log" ]; then
        count=$(wc -l < "$restart_log")
    fi
    
    if [ "$count" -ge "$MAX_RESTARTS" ]; then
        log_msg "WATCHDOG" "ERROR: Max restarts ($MAX_RESTARTS/24h) reached! Stopping watchdog."
        return 1
    fi
    
    # Log this restart
    echo "$now" >> "$restart_log"
    return 0
}

# ----------------------------
# Main loop
# ----------------------------
main() {
    log_msg "WATCHDOG" "=========================================="
    log_msg "WATCHDOG" "Watchdog starting..."
    log_msg "WATCHDOG" "  Check interval: ${CHECK_INTERVAL}s"
    log_msg "WATCHDOG" "  LLM timeout: ${LLM_TIMEOUT}s"
    log_msg "WATCHDOG" "  TG timeout: ${TG_TIMEOUT}s"
    log_msg "WATCHDOG" "  Max restarts/24h: $MAX_RESTARTS"
    log_msg "WATCHDOG" "  Grace period: ${GRACE_PERIOD}s"
    log_msg "WATCHDOG" "=========================================="
    log_json "watchdog_start" \
        "\"check_interval\": $CHECK_INTERVAL, \"llm_timeout\": $LLM_TIMEOUT, \"tg_timeout\": $TG_TIMEOUT, \"max_restarts\": $MAX_RESTARTS"
    
    while true; do
        # Check supervisor
        if ! is_supervisor_alive; then
            log_msg "WATCHDOG" "WARNING: Supervisor process not found!"
            
            if check_restart_rate; then
                restart_supervisor "process_not_found"
            else
                log_msg "WATCHDOG" "ABORTED: Too many restarts. Manual intervention required."
                exit 1
            fi
        else
            local pid
            pid=$(get_supervisor_pid)
            
            # Check heartbeat freshness
            local heartbeat_file="$LOG_DIR/supervisor.jsonl"
            if [ -f "$heartbeat_file" ]; then
                local last_heartbeat
                last_heartbeat=$(tail -1 "$heartbeat_file" 2>/dev/null | grep -o '"ts": "[^"]*"' | head -1 | cut -d'"' -f4)
                if [ -n "$last_heartbeat" ]; then
                    local last_ts
                    last_ts=$(date -d "$last_heartbeat" +%s 2>/dev/null) || last_ts=0
                    local now
                    now=$(date +%s)
                    local age=$((now - last_ts))
                    
                    if [ "$age" -gt $((CHECK_INTERVAL * 3)) ]; then
                        log_msg "WATCHDOG" "WARNING: Heartbeat stale (${age}s old). Restarting..."
                        if check_restart_rate; then
                            restart_supervisor "heartbeat_stale_age=${age}s"
                        else
                            log_msg "WATCHDOG" "ABORTED: Too many restarts."
                            exit 1
                        fi
                    fi
                fi
            fi
        fi
        
        # Check llama-server health (informational only)
        if ! check_llama_health; then
            log_msg "WATCHDOG" "WARNING: llama-server not responding on port 8080"
            log_json "health_check" "\"llama\": \"unhealthy\""
        else
            log_json "health_check" "\"llama\": \"healthy\""
        fi
        
        # Check Telegram API (informational only)
        if ! check_telegram_health; then
            log_msg "WATCHDOG" "WARNING: Telegram API not responding"
            log_json "health_check" "\"telegram\": \"unhealthy\""
        else
            log_json "health_check" "\"telegram\": \"healthy\""
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

main
