#!/usr/bin/env bash
# Run the cross-model communication matrix with plain Bash.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_INDEX="${CONFIG_INDEX:-all}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
STATE_ROOT="${STATE_ROOT:-${SCRIPT_DIR}/state/hetero}"
FORCE_ALL="${FORCE_ALL:-false}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.10}"
SEQUENTIAL_INFO_ONLY="${SEQUENTIAL_INFO_ONLY:-false}"
LATENT_ONLY="${LATENT_ONLY:-true}"

usage() {
    cat <<'EOF'
Usage: bash run_hetero.sh [--index N] [--force]

Runs all configurations sequentially by default. Use --index N (1-based) to
run one configuration. Completed configurations are skipped unless --force is
used. Any run.sh setting can be supplied as an environment variable.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --index)
            [[ $# -ge 2 ]] || { echo "ERROR: --index requires a value" >&2; exit 2; }
            CONFIG_INDEX="$2"
            shift 2
            ;;
        --force)
            FORCE_ALL=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

DATASETS=(aime2024 aime2025 gpqa humanevalplus mbppplus medqa)
SENDERS=("Qwen/Qwen3-14B" "Qwen/Qwen3-8B")
RECEIVERS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
METHODS=(text_mas latent_mas_hybrid latent_mas_hybrid latent_mas_hybrid)
ALIGNMENTS=(identical linear soft kernel)
PROMPT=sequential

DATASET_COUNT=${#DATASETS[@]}
DIRECTION_COUNT=${#SENDERS[@]}
EXPERIMENT_COUNT=${#METHODS[@]}
TOTAL_COUNT=$((DATASET_COUNT * DIRECTION_COUNT * EXPERIMENT_COUNT))

if [[ "${CONFIG_INDEX}" != all ]] && { ! [[ "${CONFIG_INDEX}" =~ ^[1-9][0-9]*$ ]] || (( CONFIG_INDEX > TOTAL_COUNT )); }; then
    echo "ERROR: --index must be between 1 and ${TOTAL_COUNT}." >&2
    exit 2
fi

mkdir -p "${STATE_ROOT}"

run_config() {
    local index="$1"
    local offset=$((index - 1))
    local experiment_index=$((offset % EXPERIMENT_COUNT))
    local dataset_direction_index=$((offset / EXPERIMENT_COUNT))
    local dataset_index=$((dataset_direction_index % DATASET_COUNT))
    local direction_index=$((dataset_direction_index / DATASET_COUNT))
    local task="${DATASETS[dataset_index]}"
    local sender="${SENDERS[direction_index]}"
    local receiver="${RECEIVERS[direction_index]}"
    local method="${METHODS[experiment_index]}"
    local alignment="${ALIGNMENTS[experiment_index]}"
    local sender_slug receiver_slug state_file

    sender_slug="$(printf '%s' "${sender}" | tr -c 'A-Za-z0-9._-' '_')"
    receiver_slug="$(printf '%s' "${receiver}" | tr -c 'A-Za-z0-9._-' '_')"
    state_file="${STATE_ROOT}/${index}_${task}_${alignment}_${sender_slug}_to_${receiver_slug}.log"

    if [[ "${FORCE_ALL}" != true ]] && [[ -f "${state_file}" ]] && [[ "$(tail -n 1 "${state_file}")" == "Exit status: 0" ]]; then
        echo "Skipping completed configuration ${index}/${TOTAL_COUNT}: ${state_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${index}/${TOTAL_COUNT}: ${task} ${sender} -> ${receiver} ${method}/${alignment}"
    env \
        FULL_EXP=false \
        TASK_ONLY=true \
        SINGLE_CONFIG=true \
        CAPTURE_ALL_OUTPUT=true \
        CONFIG_METHOD="${method}" \
        CONFIG_PROMPT="${PROMPT}" \
        CONFIG_ALIGNMENT="${alignment}" \
        MODEL_NAME="${sender}" \
        AGENT_MODELS="${sender} ${receiver}" \
        TASK="${task}" \
        MAX_SAMPLES="${MAX_SAMPLES}" \
        STATE_FILE="${state_file}" \
        REPETITION_PENALTY="${REPETITION_PENALTY}" \
        SEQUENTIAL_INFO_ONLY="${SEQUENTIAL_INFO_ONLY}" \
        LATENT_ONLY="${LATENT_ONLY}" \
        bash "${SCRIPT_DIR}/run.sh"
}

if [[ "${CONFIG_INDEX}" == all ]]; then
    for ((index = 1; index <= TOTAL_COUNT; index++)); do
        run_config "${index}"
    done
else
    run_config "${CONFIG_INDEX}"
fi

echo "Cross-model runs completed. Logs: ${STATE_ROOT}"
