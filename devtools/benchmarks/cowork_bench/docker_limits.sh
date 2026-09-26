#!/usr/bin/env bash
# Run-scoped resource flags for the pinned Cowork runner, not a general Docker
# wrapper. The runner uses plain run/exec/rm and network subcommands, with the
# daemon selected through DOCKER_HOST. Never print argv: it can contain secrets.
set -euo pipefail
: "${COWORK_REAL_DOCKER:?missing resolved Docker executable}"

# The word after an exact option, without printing any argument.
option_value() {
    local option=$1 previous=""
    shift
    for argument in "$@"; do
        if [ "$previous" = "$option" ]; then
            printf '%s' "$argument"
            return 0
        fi
        previous=$argument
    done
    return 1
}

# Host-only eval-attempt evidence (COWORK_ADMISSION_DIR is never mounted into a
# container). Evidence failures never block the runner's own command.
if [ "${1:-}" = exec ] && [ -n "${COWORK_ADMISSION_DIR:-}" ]; then
    case "${2:-}" in
        agent-*)
            # The runner names agent-, eval- and pg- containers with one task ID.
            if task=$(option_value --task_dir "$@"); then
                { mkdir -p "$COWORK_ADMISSION_DIR" \
                    && printf '%s\n' "$task" > "$COWORK_ADMISSION_DIR/${2#agent-}.task"; } 2>/dev/null || true
            fi ;;
        eval-*)
            if [ "$(option_value --phase "$@" || true)" = eval ]; then
                # The runner ran `rm -fv agent-... || true`; only a daemon listing
                # without that exact name proves the agent container is gone.
                agent="agent-${2#eval-}"
                state=unknown
                if names=$("$COWORK_REAL_DOCKER" ps -a --filter "name=$agent" --format '{{.Names}}' 2>/dev/null); then
                    state=absent
                    while IFS= read -r name; do
                        if [ "$name" = "$agent" ]; then state=present; fi
                    done <<< "$names"
                fi
                shift
                exec "$COWORK_REAL_DOCKER" exec -e "COWORK_AGENT_CONTAINER_STATE=$state" "$@"
            fi ;;
    esac
fi

create_kind=""
case "${1:-}" in
    run|create) create_kind=container; prefix=("$1"); shift ;;
    container)
        case "${2:-}" in
            run|create) create_kind=container; prefix=("$1" "$2"); shift 2 ;;
        esac ;;
    network)
        if [ "${2:-}" = create ]; then
            create_kind=network; prefix=("$1" "$2"); shift 2
        fi ;;
esac
if [ -z "$create_kind" ]; then
    exec "$COWORK_REAL_DOCKER" "$@"
fi

: "${COWORK_RUN_LABEL:?missing exact run label}"
: "${COWORK_RESOURCE_ROOT:?missing resource filesystem}"
: "${COWORK_MIN_FREE_BYTES:?missing disk reserve}"
: "${COWORK_STOP_FILE:?missing admission stop file}"
container_name=""
if [ "$create_kind" = container ]; then
    container_name=$(option_value --name "$@" || true)
fi

refuse() {
    # The first cause survives later attempts, including a parent budget stop.
    (set -C; printf '%s\n' "$1" > "$COWORK_STOP_FILE") 2>/dev/null || true
    # A refused eval container proves this task's evaluator never started.
    if [ -n "${COWORK_ADMISSION_DIR:-}" ] && [[ "$container_name" == eval-* ]]; then
        { mkdir -p "$COWORK_ADMISSION_DIR" \
            && (set -C; printf '%s\n' "$1" > "$COWORK_ADMISSION_DIR/${container_name#eval-}.eval_refused"); } \
            2>/dev/null || true
    fi
    printf 'Cowork resource admission stopped: %s\n' "$1" >&2
    exit 75
}
if [ -e "$COWORK_STOP_FILE" ]; then
    refuse admission_stopped
fi
if ! free_kib=$(LC_ALL=C df -Pk "$COWORK_RESOURCE_ROOT" | awk 'NR == 2 {print $4}'); then
    refuse disk_probe_failed
fi
if ! [[ "$free_kib" =~ ^[0-9]+$ && "$COWORK_MIN_FREE_BYTES" =~ ^[0-9]+$ ]]; then
    refuse disk_probe_invalid
fi
if (( free_kib * 1024 < COWORK_MIN_FREE_BYTES )); then
    refuse disk_reserve_reached
fi

label_args=(--label "org.ouroboros.cowork.run=$COWORK_RUN_LABEL")
if [ "$create_kind" = network ]; then
    exec "$COWORK_REAL_DOCKER" "${prefix[@]}" "${label_args[@]}" "$@"
fi
: "${COWORK_CONTAINER_CPUS:?missing CPU bound}"
: "${COWORK_CONTAINER_MEMORY:?missing memory bound}"
: "${COWORK_CONTAINER_PIDS:?missing PID bound}"
# Equal memory and memory-swap forbids swap use, as in the SWE-Pro adapter.
exec "$COWORK_REAL_DOCKER" "${prefix[@]}" \
    --cpus "$COWORK_CONTAINER_CPUS" \
    --memory "$COWORK_CONTAINER_MEMORY" --memory-swap "$COWORK_CONTAINER_MEMORY" \
    --pids-limit "$COWORK_CONTAINER_PIDS" --pull=never \
    "${label_args[@]}" "$@"
