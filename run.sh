#!/usr/bin/env bash
# Generic Bash launcher. It has no scheduler, module-system, filesystem, or
# machine-specific assumptions.

set -euo pipefail

FULL_EXP="${FULL_EXP:-false}"
TASK_ONLY="${TASK_ONLY:-false}"
STATE_FILE="${STATE_FILE:-state/run_state.txt}"
RUN_SCRIPT="${RUN_SCRIPT:-run.py}"
RESULT_ROOT="${RESULT_ROOT:-result}"
LOG_ROOT="${LOG_ROOT:-logging}"
SINGLE_CONFIG="${SINGLE_CONFIG:-false}"
CONFIG_METHOD="${CONFIG_METHOD:-}"
CONFIG_PROMPT="${CONFIG_PROMPT:-}"
CONFIG_ALIGNMENT="${CONFIG_ALIGNMENT:-identical}"
AGENT_MODELS="${AGENT_MODELS:-}"
CAPTURE_ALL_OUTPUT="${CAPTURE_ALL_OUTPUT:-false}"
RUN_OUTPUT_WRAPPED="${RUN_OUTPUT_WRAPPED:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
for ARG in "$@"; do
    case "${ARG}" in
        --full_exp) FULL_EXP=true ;;
        *)
            echo "ERROR: Unknown argument: ${ARG}"
            echo "Usage: bash run.sh [--full_exp]"
            exit 2
            ;;
    esac
done

if [ "${SINGLE_CONFIG}" = true ]; then
    case "${CONFIG_METHOD}" in
        baseline|text_mas)
            if [ "${CONFIG_ALIGNMENT}" != "identical" ]; then
                echo "ERROR: ${CONFIG_METHOD} only supports identical alignment."
                exit 2
            fi
            ;;
        latent_mas|latent_mas_hybrid)
            case "${CONFIG_ALIGNMENT}" in
                identical|linear|kernel|kernel_early_stopping|soft|text) ;;
                *) echo "ERROR: invalid CONFIG_ALIGNMENT=${CONFIG_ALIGNMENT}"; exit 2 ;;
            esac
            ;;
        *) echo "ERROR: invalid CONFIG_METHOD=${CONFIG_METHOD}"; exit 2 ;;
    esac
    case "${CONFIG_PROMPT}" in
        sequential|hierarchical) ;;
        *) echo "ERROR: invalid CONFIG_PROMPT=${CONFIG_PROMPT}"; exit 2 ;;
    esac
fi

# Batch launchers can request an outer wrapper so stdout and stderr land in the
# configuration's state file. Validation runs before that file is created.
if [ "${CAPTURE_ALL_OUTPUT}" = true ] && [ "${RUN_OUTPUT_WRAPPED}" != true ]; then
    mkdir -p "$(dirname "${STATE_FILE}")"
    export FULL_EXP TASK_ONLY STATE_FILE RUN_SCRIPT RESULT_ROOT LOG_ROOT
    export SINGLE_CONFIG CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT AGENT_MODELS
    export GENERATE_BS GENERATE_BS_DIVISOR SEED TIMES
    export KERNEL_GATE_MODE KERNEL_FIXED_FEATURE KERNEL_TOPK
    export CAPTURE_ALL_OUTPUT RUN_OUTPUT_WRAPPED=true
    exec bash "${BASH_SOURCE[0]}" "$@" > "${STATE_FILE}" 2>&1
fi

## ========================== Environment =====================================
cd "${SCRIPT_DIR}" || exit 1
if [ -f "${VENV_PATH}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
elif [ -f "${VENV_PATH}/Scripts/activate" ]; then
    # Git Bash/WSL-compatible activation for a Windows-created environment.
    # shellcheck disable=SC1091
    source "${VENV_PATH}/Scripts/activate"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 127
fi
if [ ! -f "${RUN_SCRIPT}" ]; then
    echo "ERROR: ${RUN_SCRIPT} not found in ${SCRIPT_DIR}." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1

## ========================== Experiment Config ================================
## Keep named values here so the launch command and job summary stay in sync.

## --- Core dataset / model settings ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}" # Hugging Face model ID passed to --model_name.
TASK="${TASK:-humanevalplus}"              # Evaluation dataset/task name.
PROMPT_SEQUENTIAL="sequential"      # Sequential multi-agent architecture.
PROMPT_HIERARCHICAL="hierarchical"  # Hierarchical multi-agent architecture.
MAX_SAMPLES="${MAX_SAMPLES:-30}" # Number of examples; -1 evaluates all examples.
SPLIT="${SPLIT:-test}"       # Dataset split requested from the task loader.
DEVICE="${DEVICE:-cuda}"     # PyTorch device used by the HF backend.

## --- Generation settings ---
# Empty means use params_dict.json[TASK].max_token; an absent/unknown task
# falls back to 20000. Set a number here to explicitly override every task.
MAX_NEW_TOKENS=""
resolve_max_new_tokens() {
    if [ -n "${MAX_NEW_TOKENS}" ]; then
        printf '%s\n' "${MAX_NEW_TOKENS}"
        return
    fi

    "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

fallback = 20000
try:
    params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
    task_params = params.get(sys.argv[1], {})
    value = task_params.get("max_token", fallback) if isinstance(task_params, dict) else fallback
    print(value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else fallback)
except (OSError, json.JSONDecodeError):
    print(fallback)
PY
}
TEMPERATURE="${TEMPERATURE:-0.6}" # Sampling temperature; model wrappers may override.
TOP_P="${TOP_P:-0.95}"             # Nucleus-sampling threshold; model wrappers may override.
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}" # 1.0 preserves the original generation behavior.
# Empty means use params_dict.json[TASK].generation_bs; fallback: 10.
GENERATE_BS="${GENERATE_BS:-}"
GENERATE_BS_DIVISOR="${GENERATE_BS_DIVISOR:-1}"
if ! [[ "${GENERATE_BS_DIVISOR}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GENERATE_BS_DIVISOR must be a positive integer, got: ${GENERATE_BS_DIVISOR}"
    exit 2
fi
resolve_generate_bs() {
    if [ -n "${GENERATE_BS}" ]; then
        printf '%s\n' "${GENERATE_BS}"
        return
    fi

    "${PYTHON_BIN}" - "$1" "${GENERATE_BS_DIVISOR}" <<'PY'
import json
import sys
from pathlib import Path

fallback = 10
try:
    params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
    task_params = params.get(sys.argv[1], {})
    value = task_params.get("generation_bs", fallback) if isinstance(task_params, dict) else fallback
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        value = fallback
    divisor = int(sys.argv[2])
    print(max(1, value // divisor))
except (OSError, json.JSONDecodeError):
    print(max(1, fallback // int(sys.argv[2])))
PY
}
SEED="${SEED:-42}"     # Random seed for reproducibility.
# Empty means use params_dict.json[TASK].times; fallback: 1. Each repetition
# uses SEED, SEED+1, ... and is included in a per-configuration average JSON.
TIMES="${TIMES:-}"
resolve_times() {
    if [ -n "${TIMES}" ]; then
        printf '%s\n' "${TIMES}"
        return
    fi

    "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

fallback = 1
try:
    params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
    task_params = params.get(sys.argv[1], {})
    value = task_params.get("times", fallback) if isinstance(task_params, dict) else fallback
    print(value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else fallback)
except (OSError, json.JSONDecodeError):
    print(fallback)
PY
}

## --- Text and latent communication settings ---
TEXT_MAS_CONTEXT_LENGTH=-1  # TextMAS context limit; -1 means unlimited.
# Empty means use params_dict.json[TASK].latent_steps[PROMPT]; fallback: 20.
LATENT_STEPS="${LATENT_STEPS:-}"
resolve_latent_steps() {
    if [ -n "${LATENT_STEPS}" ]; then
        printf '%s\n' "${LATENT_STEPS}"
        return
    fi

    "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

fallback = 20
try:
    params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
    task_params = params.get(sys.argv[1], {})
    latent_steps = task_params.get("latent_steps", {}) if isinstance(task_params, dict) else {}
    value = latent_steps.get(sys.argv[2], fallback) if isinstance(latent_steps, dict) else fallback
    print(value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else fallback)
except (OSError, json.JSONDecodeError):
    print(fallback)
PY
}
TRUST_REMOTE_CODE=true      # Pass --trust_remote_code when the model requires it.
SEQUENTIAL_INFO_ONLY="${SEQUENTIAL_INFO_ONLY:-false}" # Retain only each agent's own prompt + latent KV before the next agent.
LATENT_ONLY="${LATENT_ONLY:-false}"                   # Retain only latent KV before the next agent (implies SEQUENTIAL_INFO_ONLY).
THINK="${THINK:-auto}"      # auto uses the reasoning-model registry in run.py.

## --- Alignment settings ---
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"                 # Linear ridge.
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"         # Random-feature count.
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"   # Kernel temperature.
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"    # Kernel chunk size.
KERNEL_GATE_MODE="${KERNEL_GATE_MODE:-soft}"      # soft, argmax, fixed, or topk.
KERNEL_FIXED_FEATURE="${KERNEL_FIXED_FEATURE:-0}" # Feature used by fixed gate.
KERNEL_TOPK="${KERNEL_TOPK:-8}"                   # Features retained by top-k gate.
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"       # Soft Thinking logits temperature.
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"          # Hidden queries per softmax chunk.
SOFT_TOP_K="${SOFT_TOP_K:-30}"                    # Reference top-k pre-filter.
SOFT_TOP_P="${SOFT_TOP_P:-0.95}"                  # Reference nucleus threshold.
SOFT_MIN_P="${SOFT_MIN_P:-0.001}"                 # Probability floor relative to max.
SOFT_MAX_TOPK="${SOFT_MAX_TOPK:-10}"              # Final concept-token support size.
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"
EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"

## --- vLLM backend settings ---
USE_VLLM=false              # Whether to enable the optional vLLM backend.
TENSOR_PARALLEL_SIZE=1      # Number of GPUs used for vLLM tensor parallelism.
GPU_MEMORY_UTILIZATION=0.9  # Fraction of each GPU vLLM may reserve.

build_common_args() {
local prompt="$1"
local run_seed="${2:-${SEED}}"
RESOLVED_LATENT_STEPS="$(resolve_latent_steps "${TASK}" "${prompt}")"
COMMON=(
    # Core dataset / model settings
    --model_name "${MODEL_NAME}"             # Required model ID
    --task "${TASK}"                         # run.py default: humanevalplus
    --max_samples "${MAX_SAMPLES}"           # run.py default: -1 (all samples)
    --split "${SPLIT}"                       # run.py default: test; AIME always uses train
    --device "${DEVICE}"                     # run.py default: cuda

    # Generation settings
    --max_new_tokens "${RESOLVED_MAX_NEW_TOKENS}" # Task default from params_dict.json; fallback: 20000
    --temperature "${TEMPERATURE}"           # run.py default: 0.6
    --top_p "${TOP_P}"                       # run.py default: 0.95
    --repetition_penalty "${REPETITION_PENALTY}" # run.py default: 1.0 (disabled)
    --generate_bs "${RESOLVED_GENERATE_BS}"  # Task default from params_dict.json; fallback: 10
    --seed "${run_seed}"                     # Base seed plus repetition index

    # Text and latent communication settings
    --text_mas_context_length "${TEXT_MAS_CONTEXT_LENGTH}" # run.py default: -1 (unlimited)
    --latent_steps "${RESOLVED_LATENT_STEPS}" # Task/prompt default from params_dict.json; fallback: 20

    # Alignment settings. --align_method is varied per latent-state command below.
    --align_ridge "${ALIGN_RIDGE}"           # run.py default: 1e-5; used by linear
    --kernel_features "${KERNEL_FEATURES}"   # run.py default: 1024; used by kernel
    --kernel_temperature "${KERNEL_TEMPERATURE}" # run.py default: 0.6; used by kernel
    --kernel_chunk_size "${KERNEL_CHUNK_SIZE}" # run.py default: 4096; used by kernel
    --kernel_gate_mode "${KERNEL_GATE_MODE}"
    --kernel_fixed_feature "${KERNEL_FIXED_FEATURE}"
    --kernel_topk "${KERNEL_TOPK}"
    --soft_temperature "${SOFT_TEMPERATURE}" # run.py default: 0.6; used by soft
    --soft_chunk_size "${SOFT_CHUNK_SIZE}" # run.py default: 32; used by soft
    --soft_top_k "${SOFT_TOP_K}"
    --soft_top_p "${SOFT_TOP_P}"
    --soft_min_p "${SOFT_MIN_P}"
    --soft_max_topk "${SOFT_MAX_TOPK}"

    # vLLM numeric settings; ignored unless --use_vllm is enabled below.
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" # run.py default: 1
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" # run.py default: 0.9
)

# Qwen3-8B requires custom generation code; run.py default is disabled.
if [ "${TRUST_REMOTE_CODE}" = true ]; then
    COMMON+=(--trust_remote_code)
fi
}

## Boolean / optional arguments for latent-state methods.
# LATENT_ONLY implies
# SEQUENTIAL_INFO_ONLY in the Python implementation, so both flags may safely
# be passed when both variables are true.
LATENT_CACHE_ARGS=()
case "${THINK}" in
    auto) ;;
    true) LATENT_CACHE_ARGS+=(--think) ;;
    false) LATENT_CACHE_ARGS+=(--no-think) ;;
    *)
        echo "ERROR: THINK must be auto, true, or false, got: ${THINK}" >&2
        exit 2
        ;;
esac
if [ "${SEQUENTIAL_INFO_ONLY}" = true ]; then
    LATENT_CACHE_ARGS+=(--sequential_info_only)
fi
if [ "${LATENT_ONLY}" = true ]; then
    LATENT_CACHE_ARGS+=(--latent_only)
fi
##   --think/--no-think       default: auto; reasoning models rely on their chat template.
##   --kernel_seed SEED       default: None; omitted uses --seed.
##   --use_vllm               default: false; enables vLLM backend.
##   --sequential_info_only   default: false; preserve only the current agent's prompt + latent KV cache.
##   --latent_only            default: false; preserve only latent KV cache; implies --sequential_info_only.
##   --enable_prefix_caching  default: false; only relevant with vLLM.
##   --use_second_HF_model    default: false; only relevant with latent_mas + vLLM.
##   --device2 DEVICE         default: None, then run.py uses --device.
##   --agent_models MODEL...  default: None; used by heterogeneous text_mas and latent_mas_hybrid.
##
## --align_method choices/default: identical (default), linear, kernel, kernel_early_stopping, soft, text.
## The current suite runs all five methods explicitly below.

## ========================== Run Experiment Suite =============================
## Store each repetition separately and write one average summary per config.
run_repeated() {
    local method="$1"
    local prompt="$2"
    local align_method="$3"
    local method_slug="${method}"
    local config_name result_dir log_dir
    local repeat_index run_seed result_path log_path
    local -a result_paths command

    # Alignment is part of the effective latent-state method. Without it, the
    # identical/linear/kernel/kernel_early_stopping/soft suites would overwrite one another.
    if [[ "${method}" = "latent_mas" || "${method}" = "latent_mas_hybrid" ]]; then
        method_slug="${method}_${align_method}"
        if [[ "${align_method}" = "kernel" && "${KERNEL_GATE_MODE}" != "soft" ]]; then
            method_slug="${method_slug}_${KERNEL_GATE_MODE}"
            if [[ "${KERNEL_GATE_MODE}" = "fixed" ]]; then
                method_slug="${method_slug}_${KERNEL_FIXED_FEATURE}"
            elif [[ "${KERNEL_GATE_MODE}" = "topk" ]]; then
                method_slug="${method_slug}_${KERNEL_TOPK}"
            fi
        fi
    fi
    config_name="${TASK}_${method_slug}_${prompt}_${MODEL_SLUG}_${RUN_TIME}"
    result_dir="${RESULT_ROOT}/${config_name}"
    log_dir="${LOG_ROOT}/${config_name}"
    mkdir -p "${result_dir}" "${log_dir}"

    result_paths=()
    for ((repeat_index = 1; repeat_index <= RESOLVED_TIMES; repeat_index++)); do
        run_seed=$((SEED + repeat_index - 1))
        result_path="${result_dir}/repeat_${repeat_index}.json"
        log_path="${log_dir}/repeat_${repeat_index}.txt"
        result_paths+=("${result_path}")
        build_common_args "${prompt}" "${run_seed}"
        : > "${log_path}"

        echo "Running ${method}/${prompt}/${align_method}, repeat ${repeat_index}/${RESOLVED_TIMES}, seed=${run_seed}"
        command=("${PYTHON_BIN}" "${RUN_SCRIPT}" --method "${method}" --prompt "${prompt}" \
            --result_path "${result_path}" --log_path "${log_path}" "${COMMON[@]}")
        if [[ "${method}" = "latent_mas_hybrid" && -z "${AGENT_MODELS}" ]]; then
            echo "ERROR: AGENT_MODELS is required for latent_mas_hybrid." >&2
            return 2
        fi
        if [[ ("${method}" = "text_mas" || "${method}" = "latent_mas_hybrid") && -n "${AGENT_MODELS}" ]]; then
            read -r -a HETERO_AGENT_MODELS <<< "${AGENT_MODELS}"
            if (( ${#HETERO_AGENT_MODELS[@]} != 2 && ${#HETERO_AGENT_MODELS[@]} != 4 )); then
                echo "ERROR: AGENT_MODELS must contain either two or four model IDs." >&2
                return 2
            fi
            command+=(--agent_models "${HETERO_AGENT_MODELS[@]}")
        fi
        if [[ "${method}" = "latent_mas" || "${method}" = "latent_mas_hybrid" ]]; then
            command+=(--align_method "${align_method}" "${LATENT_CACHE_ARGS[@]}")
            if [ "${EARLY_STOPPING_LENGTH_THRESHOLD}" != "auto" ]; then
                command+=(--early_stopping_length_threshold "${EARLY_STOPPING_LENGTH_THRESHOLD}")
            fi
            if [ "${EARLY_STOPPING_ENTROPY_THRESHOLD}" != "auto" ]; then
                command+=(--early_stopping_entropy_threshold "${EARLY_STOPPING_ENTROPY_THRESHOLD}")
            fi
        fi
        "${command[@]}" || return $?
    done

    "${PYTHON_BIN}" -m src.utils aggregate-results --output "${result_dir}/summary.json" \
        --expected-repetitions "${RESOLVED_TIMES}" "${result_paths[@]}"
}

## All experiment output is recorded in the task state file. A failed repeat
## stops the Bash suite with a nonzero exit status.
run_suite() {
    local suite_model="$1"
    local suite_task="$2"
    MODEL_NAME="${suite_model}"
    TASK="${suite_task}"
    RESOLVED_MAX_NEW_TOKENS="$(resolve_max_new_tokens "${TASK}")"
    RESOLVED_GENERATE_BS="$(resolve_generate_bs "${TASK}")"
    RESOLVED_SEQUENTIAL_LATENT_STEPS="$(resolve_latent_steps "${TASK}" "${PROMPT_SEQUENTIAL}")"
    RESOLVED_HIERARCHICAL_LATENT_STEPS="$(resolve_latent_steps "${TASK}" "${PROMPT_HIERARCHICAL}")"
    RESOLVED_TIMES="$(resolve_times "${TASK}")"
    MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"

    echo "========================================================================"
    echo "  Run ID       : ${RUN_TIME}"
    echo "  Full exp     : ${FULL_EXP}"
    echo "  Model        : ${MODEL_NAME}"
    echo "  Task         : ${TASK}"
    echo "  Latent prompts: ${PROMPT_SEQUENTIAL}, ${PROMPT_HIERARCHICAL}"
    echo "  Generate BS  : ${RESOLVED_GENERATE_BS}"
    echo "  Max samples  : ${MAX_SAMPLES}"
    echo "  Repetitions  : ${RESOLVED_TIMES} (seeds ${SEED}..$((SEED + RESOLVED_TIMES - 1)))"
    echo "  Max tokens   : ${RESOLVED_MAX_NEW_TOKENS}"
    echo "  Repetition penalty: ${REPETITION_PENALTY}"
    echo "  Latent steps : ${PROMPT_SEQUENTIAL}=${RESOLVED_SEQUENTIAL_LATENT_STEPS}, ${PROMPT_HIERARCHICAL}=${RESOLVED_HIERARCHICAL_LATENT_STEPS}"
    echo "  vLLM         : ${USE_VLLM}"
    echo "  CUDA devices : ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "  Sequential info only: ${SEQUENTIAL_INFO_ONLY}"
    echo "  Latent only  : ${LATENT_ONLY}"
    echo "  Think token  : ${THINK}"
    echo "  Single config: ${SINGLE_CONFIG} ${CONFIG_METHOD}/${CONFIG_PROMPT}/${CONFIG_ALIGNMENT}"
    echo "========================================================================"

    if [ "${SINGLE_CONFIG}" = true ]; then
        run_repeated "${CONFIG_METHOD}" "${CONFIG_PROMPT}" "${CONFIG_ALIGNMENT}"
        return $?
    fi

    run_repeated baseline "${PROMPT_SEQUENTIAL}" identical &&
    run_repeated baseline "${PROMPT_HIERARCHICAL}" identical &&
    run_repeated text_mas "${PROMPT_SEQUENTIAL}" identical &&
    run_repeated text_mas "${PROMPT_HIERARCHICAL}" identical &&
    run_repeated latent_mas "${PROMPT_SEQUENTIAL}" identical &&
    run_repeated latent_mas "${PROMPT_SEQUENTIAL}" linear &&
    run_repeated latent_mas "${PROMPT_SEQUENTIAL}" kernel &&
    run_repeated latent_mas "${PROMPT_SEQUENTIAL}" kernel_early_stopping &&
    run_repeated latent_mas "${PROMPT_SEQUENTIAL}" soft &&
    run_repeated latent_mas "${PROMPT_HIERARCHICAL}" identical &&
    run_repeated latent_mas "${PROMPT_HIERARCHICAL}" linear &&
    run_repeated latent_mas "${PROMPT_HIERARCHICAL}" kernel &&
    run_repeated latent_mas "${PROMPT_HIERARCHICAL}" kernel_early_stopping &&
    run_repeated latent_mas "${PROMPT_HIERARCHICAL}" soft
}
if [ "${FULL_EXP}" = true ]; then
    MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-4B")
    if [ "${TASK_ONLY}" = true ]; then
        TASKS=("${TASK}")
    else
        mapfile -t TASKS < <("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
for task in params:
    print(task)
PY
)
    fi
else
    MODELS=("${MODEL_NAME}")
    TASKS=("${TASK}")
fi

RUN_TIME="${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}"
RUN_TIME="$(printf '%s' "${RUN_TIME}" | tr -c 'A-Za-z0-9._-' '_')"
mkdir -p "$(dirname "${STATE_FILE}")"
run_main() {
    STATUS=0
    for CURRENT_MODEL in "${MODELS[@]}"; do
        for CURRENT_TASK in "${TASKS[@]}"; do
            if [ "${CURRENT_MODEL}" = "Qwen/Qwen3-4B" ]; then
                case "${CURRENT_TASK}" in
                    aime2024|aime2025|gpqa)
                        echo "Skipped unsupported model/task combination: model=${CURRENT_MODEL}, task=${CURRENT_TASK}"
                        continue
                        ;;
                esac
            fi
            run_suite "${CURRENT_MODEL}" "${CURRENT_TASK}" || {
                STATUS=$?
                echo "ERROR: Suite failed for model=${CURRENT_MODEL}, task=${CURRENT_TASK}"
                break 2
            }
        done
    done

    echo ""
    echo "Finished at: $(date)"
    echo "Exit status: ${STATUS}"
    return "${STATUS}"
}

if [ "${RUN_OUTPUT_WRAPPED}" = true ]; then
    run_main
else
    run_main > "${STATE_FILE}" 2>&1
fi
