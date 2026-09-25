#!/usr/bin/env bash
# Run the homogeneous Exact-Soft Align reference configurations with plain Bash.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_INDEX="${CONFIG_INDEX:-all}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
STATE_ROOT="${STATE_ROOT:-${SCRIPT_DIR}/state/homo}"

usage() {
    cat <<'EOF'
Usage: bash run_homo.sh [--index N]

Runs all configurations sequentially by default. Use --index N (1-based) to
run only one configuration. Any run.sh setting can be supplied as an
environment variable, for example MAX_SAMPLES=5 TIMES=1.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --index)
            [[ $# -ge 2 ]] || { echo "ERROR: --index requires a value" >&2; exit 2; }
            CONFIG_INDEX="$2"
            shift 2
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

# model|dataset|prompt
EXPERIMENTS=(
    "Qwen/Qwen3-8B|aime2024|hierarchical"
    "Qwen/Qwen3-8B|humanevalplus|hierarchical"
    "Qwen/Qwen3-8B|gpqa|hierarchical"
    "Qwen/Qwen3-8B|medqa|hierarchical"
    "Qwen/Qwen3-14B|mbppplus|sequential"
    "Qwen/Qwen3-14B|mbppplus|hierarchical"
)

TOTAL_COUNT=${#EXPERIMENTS[@]}
if [[ "${CONFIG_INDEX}" != all ]] && { ! [[ "${CONFIG_INDEX}" =~ ^[1-9][0-9]*$ ]] || (( CONFIG_INDEX > TOTAL_COUNT )); }; then
    echo "ERROR: --index must be between 1 and ${TOTAL_COUNT}." >&2
    exit 2
fi

mkdir -p "${STATE_ROOT}"

run_config() {
    local index="$1"
    local entry="${EXPERIMENTS[index - 1]}"
    local model task prompt model_slug state_file

    IFS='|' read -r model task prompt <<< "${entry}"
    model_slug="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9._-' '_')"
    state_file="${STATE_ROOT}/${index}_${task}_latent_mas_soft_${prompt}_${model_slug}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${index}/${TOTAL_COUNT}: ${task} ${model} ${prompt} soft"
    env \
        FULL_EXP=false \
        TASK_ONLY=true \
        SINGLE_CONFIG=true \
        CAPTURE_ALL_OUTPUT=true \
        CONFIG_METHOD=latent_mas \
        CONFIG_PROMPT="${prompt}" \
        CONFIG_ALIGNMENT=soft \
        MODEL_NAME="${model}" \
        TASK="${task}" \
        MAX_SAMPLES="${MAX_SAMPLES}" \
        STATE_FILE="${state_file}" \
        bash "${SCRIPT_DIR}/run.sh"
}

if [[ "${CONFIG_INDEX}" == all ]]; then
    for ((index = 1; index <= TOTAL_COUNT; index++)); do
        run_config "${index}"
    done
else
    run_config "${CONFIG_INDEX}"
fi

echo "Homogeneous runs completed. Logs: ${STATE_ROOT}"
