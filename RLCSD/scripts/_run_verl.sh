#!/bin/bash
set -eo pipefail

# Base script: reads all params from a YAML config and launches verl training.
# Usage: bash scripts/_run_verl.sh <config.yaml> [extra hydra overrides...]

CONFIG=${1:?Usage: $0 <config.yaml> [extra overrides...]}
shift  # remaining args are extra hydra overrides

PYTHON_BIN="${PYTHON_BIN:-${PY:-python3}}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-""}
export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-12.6"}
export PATH="$CUDA_HOME/bin:$PATH"
if [ -n "${HF_ENDPOINT:-}" ]; then export HF_ENDPOINT; fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}
# Note: do NOT set TORCH_DISTRIBUTED_DEBUG=DETAIL — its debug PG wrapper
# lacks `allgather_into_tensor_coalesced`, which FSDP2/DTensor need when
# resharding weights to vLLM. NCCL flight recorder above is enough to
# capture stack traces on collective desync.

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/third_party/verl:$(pwd):${PYTHONPATH:-}"

# --- Read YAML helper ---
Y() {
    "$PYTHON_BIN" -c 'import json, sys, yaml
with open(sys.argv[1]) as stream:
    value = yaml.safe_load(stream).get(sys.argv[2], sys.argv[3])
print(json.dumps(value) if isinstance(value, list) else value)' "$CONFIG" "$1" "${2:-}"
}

METHOD=$(Y method grpo)
MODEL_PATH=$(Y model_path Qwen/Qwen3-1.7B)
USE_LORA=$(Y use_lora true)
LORA_R=$(Y lora_r 64)
LORA_ALPHA=$(Y lora_alpha 128)
LR=$(Y learning_rate 1e-5)
NUM_EPOCHS=$(Y num_epochs 30)
BS=$(Y per_device_batch_size 4)
GROUP_SIZE=$(Y group_size 4)
MAX_GRAD_NORM=$(Y max_grad_norm 1.0)
WARMUP=$(Y warmup_steps 50)
WEIGHT_DECAY=$(Y weight_decay 0.01)
MAX_TRAIN=$(Y max_train_samples 17000)
SAVE_STEPS=$(Y save_steps 200)
EVAL_STEPS=$(Y eval_steps 100)
MAX_PROMPT=$(Y max_prompt_length 2048)
MAX_RESP=$(Y max_completion_length 2048)
VAL_MAX_RESP=$(Y val_max_completion_length 38912)
VAL_MAX_SAMPLES=$(Y val_max_samples -1)
FILTER_OVERLONG_PROMPTS=$(Y filter_overlong_prompts true)
DATA_TRUNCATION=$(Y truncation error)
ACTOR_MAX_TOKEN_LEN=$(Y actor_max_token_len_per_gpu "")
TEMP=$(Y temperature 1.0)
TOP_P=$(Y top_p 0.95)
TOP_K=$(Y top_k_sampling 20)
GPU_MEM=$(Y vllm_gpu_memory_utilization 0.6)
TP=$(Y vllm_tensor_parallel_size 2)
STUDENT_ENABLE_THINKING=$(Y student_enable_thinking false)
FORCE_THINK_PREFIX=$(Y force_think_prefix false)
VAL_ENABLE_THINKING=$(Y val_enable_thinking true)
VAL_DO_SAMPLE=$(Y val_do_sample true)
VAL_N=$(Y val_n 12)
VAL_TEMP=$(Y val_temperature 0.6)
VAL_TOP_P=$(Y val_top_p ${TOP_P})
VAL_TOP_K=$(Y val_top_k ${TOP_K})
VAL_BATCH_SIZE=$(Y val_batch_size 16)
TEACHER_ENABLE_THINKING=$(Y teacher_enable_thinking true)
THINKING_SYSTEM_PROMPT=$(Y thinking_system_prompt false)
PROJECT=$(Y project_name rlcsd)
EXPERIMENT_NAME=$(Y experiment_name ${METHOD})
OUTPUT_DIR=$(Y output_dir "")
VAL_BEFORE=$(Y val_before_train false)
USE_SWANLAB=$(Y use_swanlab true)
PRIVILEGED_TEXT_MODE=$(Y privileged_text_mode solution_answer)
MATH_PRIVILEGED_TEXT_MODE=$(Y math_privileged_text_mode solution_answer)
# OPD (plain on-policy distillation) keys; only consumed when METHOD=opd.
TEACHER_MODEL_PATH=$(Y teacher_model_path "")
TEACHER_ENABLE_RESOURCE_POOL=$(Y teacher_enable_resource_pool false)
TEACHER_N_GPUS_PER_NODE=$(Y teacher_n_gpus_per_node 0)
N_GPUS_PER_NODE=$(Y n_gpus_per_node 8)
ULYSSES_SP=$(Y ulysses_sequence_parallel_size 1)
TEACHER_TP_SIZE=$(Y teacher_tp_size 1)
TEACHER_GPU_MEM=$(Y teacher_gpu_memory_utilization 0.3)
DISTILL_LOSS_MODE=$(Y distillation_loss_mode k1)
DISTILL_USE_PG=$(Y distillation_use_policy_gradient true)
DISTILL_USE_TASK_REWARDS=$(Y distillation_use_task_rewards false)
DISTILL_TOPK=$(Y distillation_topk 64)
DISTILL_LOSS_COEF=$(Y distillation_loss_coef 1.0)
DISTILL_LOSS_MAX_CLAMP=$(Y distillation_loss_max_clamp 10.0)
DISTILL_LOG_PROB_MIN_CLAMP=$(Y distillation_log_prob_min_clamp -10.0)
DISTILL_NUM_WORKERS=$(Y distillation_num_workers 8)
DATA_ROOT=$(Y data_dir ./data)
DATA_SHUFFLE=$(Y data_shuffle false)
TRAIN_DATASET=$(Y train_dataset dapo_math_17k)
VAL_DATASET=$(Y val_dataset amc23,aime24,aime25)
TRAIN_FILES=$(Y train_files "")
VAL_FILES=$(Y val_files "")
REWARD_MANAGER_NAME=$(Y reward_manager_name "")
USE_CUSTOM_REWARD_FUNCTION=$(Y use_custom_reward_function true)
CUSTOM_REWARD_FUNCTION_PATH=$(Y custom_reward_function_path src/verl_reward.py)
CUSTOM_REWARD_FUNCTION_NAME=$(Y custom_reward_function_name compute_score)
GOLD_CONTEXT_FALLBACK_MODE=$(Y gold_context_fallback_mode error)
NCCL_TIMEOUT=$(Y nccl_timeout 7200)
ACTOR_PARAM_OFFLOAD=$(Y actor_param_offload false)
# 2026-08-05: repro parity knobs. Historical RLCSD runs hardcoded fsdp2 +
# bf16 MASTER weights; upstream-verl runs use fsdp + fp32 masters (bf16
# masters quantize small Adam updates at lr~2e-6). Defaults keep the old
# behavior; the repro16kmix batch sets actor_strategy=fsdp, model_dtype=fp32.
ACTOR_STRATEGY=$(Y actor_strategy fsdp2)
MODEL_DTYPE=$(Y model_dtype bf16)
# ref is forward-only: engine defaults to bf16 inference when unset, which is
# exactly repro behavior -> keep bf16 regardless of the actor master dtype.
REF_MODEL_DTYPE=$(Y ref_model_dtype bf16)
# fp32 masters push embed_tokens to 2.18GB > the 2048MB weight-sync bucket
# (legacy-worker bucketed transfer, SD arms). 4096 clears it with margin.
UPDATE_WEIGHTS_BUCKET_MB=$(Y update_weights_bucket_megabytes 2048)
ROLLOUT_LOAD_FORMAT=$(Y rollout_load_format safetensors)
ROLLOUT_LAYERED_SUMMON=$(Y rollout_layered_summon True)
ACTOR_OPTIMIZER_OFFLOAD=$(Y actor_optimizer_offload false)
REF_PARAM_OFFLOAD=$(Y ref_param_offload true)
ACTOR_PPO_MICRO_BS=$(Y actor_ppo_micro_batch_size_per_gpu 4)
ROLLOUT_LOGPROB_MICRO_BS=$(Y rollout_log_prob_micro_batch_size_per_gpu "")
REF_LOGPROB_MICRO_BS=$(Y ref_log_prob_micro_batch_size_per_gpu "")
ROLLOUT_IS=$(Y rollout_is "")
ROLLOUT_IS_THRESHOLD=$(Y rollout_is_threshold "")
ROLLOUT_IS_BATCH_NORMALIZE=$(Y rollout_is_batch_normalize "")
ROLLOUT_RS=$(Y rollout_rs "")
ROLLOUT_RS_THRESHOLD=$(Y rollout_rs_threshold "")
OVERRIDE_MAX_POSITION_EMBEDDINGS=$(Y override_max_position_embeddings "")
ROPE_SCALING_TYPE=$(Y rope_scaling_type "")
ROPE_SCALING_FACTOR=$(Y rope_scaling_factor "")
ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS=$(Y rope_scaling_original_max_position_embeddings "")
KL_LOSS_COEF=$(Y kl_loss_coef "")
# use_kl_loss=false drops the ref-policy forward entirely (KL fully off).
# Distill methods (opsd/rlsd/...) must keep it true: the teacher plumbing rides
# on the ref/kl machinery even with kl_loss_coef=0.
USE_KL_LOSS=$(Y use_kl_loss "")
# LoongRL-style reward rejection sampling: drop zero-variance uid groups in
# place AFTER advantage computation (no re-generation; batch shrinks).
# Mutually exclusive with filter_groups_enable.
REWARD_REJECTION_SAMPLING=$(Y reward_rejection_sampling "")
# DAPO-style dynamic sampling: drop zero-variance uid groups and keep
# generating until train_batch_size informative groups are buffered.
FILTER_GROUPS_ENABLE=$(Y filter_groups_enable "")
FILTER_GROUPS_MAX_NUM_GEN_BATCHES=$(Y filter_groups_max_num_gen_batches 10)
GEN_BATCH_SIZE=$(Y gen_batch_size "")
# DAPO-style asymmetric PPO clipping (clip-higher). Optional; when unset the
# ActorConfig defaults (0.2/0.2) apply, i.e. standard symmetric clipping.
CLIP_RATIO_LOW=$(Y clip_ratio_low "")
CLIP_RATIO_HIGH=$(Y clip_ratio_high "")
RESUME_DIR=""
SWANLAB_RUN_ID_OVERRIDE=""

if [ "$TP" -le 0 ] || [ "$N_GPUS_PER_NODE" -le 0 ] || [ $((N_GPUS_PER_NODE % TP)) -ne 0 ]; then
    echo "n_gpus_per_node=${N_GPUS_PER_NODE} must be a positive multiple of vllm_tensor_parallel_size=${TP}" >&2
    exit 1
fi
if [ "$ULYSSES_SP" -le 0 ] || [ $((N_GPUS_PER_NODE % ULYSSES_SP)) -ne 0 ]; then
    echo "n_gpus_per_node=${N_GPUS_PER_NODE} must be a positive multiple of ulysses_sequence_parallel_size=${ULYSSES_SP}" >&2
    exit 1
fi
NUM_DP=$((N_GPUS_PER_NODE / TP))
ROLLOUT_BATCH=$((BS * NUM_DP))                    # prompts per rollout
ROLLOUT_SAMPLES=$((ROLLOUT_BATCH * GROUP_SIZE))    # samples per rollout

# ppo_mini_batch_size: how many samples per optimizer step.
# Default = rollout_samples (one update per rollout).
# Set smaller in yaml to get multiple updates per rollout.
PPO_MINI_BS=$(Y ppo_mini_batch_size ${ROLLOUT_BATCH})
# These token budgets are for full sequence length, so they must include prompt + response.
TRAIN_MAX_SEQ_LEN=$((MAX_PROMPT + MAX_RESP))
VAL_MAX_SEQ_LEN=$((MAX_PROMPT + VAL_MAX_RESP))
ROLLOUT_MAX_MODEL_LEN=${TRAIN_MAX_SEQ_LEN}
if [ "${VAL_MAX_SEQ_LEN}" -gt "${ROLLOUT_MAX_MODEL_LEN}" ]; then
    ROLLOUT_MAX_MODEL_LEN=${VAL_MAX_SEQ_LEN}
fi
if [ -z "${ACTOR_MAX_TOKEN_LEN}" ] || [ "${ACTOR_MAX_TOKEN_LEN}" = "None" ]; then
    ACTOR_MAX_TOKEN_LEN=${TRAIN_MAX_SEQ_LEN}
fi
LOGPROB_MAX_RESP=${MAX_RESP}
if [ "${VAL_MAX_RESP}" -gt "${LOGPROB_MAX_RESP}" ]; then
    LOGPROB_MAX_RESP=${VAL_MAX_RESP}
fi
LOGPROB_MAX_SEQ_LEN=$((MAX_PROMPT + LOGPROB_MAX_RESP))
ROLLOUT_MAX_BATCHED_TOKENS=${VAL_MAX_SEQ_LEN}

USE_SWANLAB_LOWER=$(echo "$USE_SWANLAB" | tr '[:upper:]' '[:lower:]')
USE_LORA_LOWER=$(echo "$USE_LORA" | tr '[:upper:]' '[:lower:]')
USE_CUSTOM_REWARD_FUNCTION_LOWER=$(echo "$USE_CUSTOM_REWARD_FUNCTION" | tr '[:upper:]' '[:lower:]')

case "$METHOD" in
    grpo|opd|opsd|grpo_opsd|opsd_ectr|grpo_opsd_ectr|ce_rlsd_opsd_ectr|sdpo|rlsd|eg_rlsd|ce_rlsd|ce_gate_rlsd|rlsd_ectr|srpo|rlcsd)
        ;;
    *)
        echo "Invalid method '${METHOD}' in ${CONFIG}. Allowed values: grpo, opd, opsd, grpo_opsd, opsd_ectr, grpo_opsd_ectr, ce_rlsd_opsd_ectr, sdpo, rlsd, eg_rlsd, ce_rlsd, ce_gate_rlsd, rlsd_ectr, srpo, rlcsd" >&2
        exit 1
        ;;
esac

if [ "$METHOD" = "opd" ]; then
    if [ -z "$TEACHER_MODEL_PATH" ] || [ "$TEACHER_MODEL_PATH" = "None" ]; then
        echo "method=opd requires teacher_model_path in ${CONFIG}" >&2
        exit 1
    fi
fi

if [ -z "$EXPERIMENT_NAME" ] || [ "$EXPERIMENT_NAME" = "None" ]; then
    EXPERIMENT_NAME="${METHOD}"
fi

if [ -z "$OUTPUT_DIR" ] || [ "$OUTPUT_DIR" = "None" ]; then
    OUTPUT_DIR="./outputs/${PROJECT}/${EXPERIMENT_NAME}"
fi

LOGGER='["console","tensorboard","file"]'
if [ "$USE_SWANLAB_LOWER" = "true" ]; then
    LOGGER='["console","tensorboard","file","swanlab"]'
fi

if [ "$USE_SWANLAB_LOWER" = "true" ] && [ -z "${SWANLAB_API_KEY}" ]; then
    echo "Warning: SWANLAB_API_KEY is empty; swanlab logging requires existing login state or an exported key" >&2
fi

LORA_ARGS=(
    actor_rollout_ref.model.lora.merge=False
    actor_rollout_ref.model.lora_rank=0
)
if [ "$USE_LORA_LOWER" = "true" ]; then
    LORA_ARGS=(
        actor_rollout_ref.model.lora.merge=True
        actor_rollout_ref.model.lora_rank=${LORA_R}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    )
fi

HYDRA_ARGS=()
TOTAL_STEPS=$(Y total_training_steps)
if [[ -n "$TOTAL_STEPS" && "$TOTAL_STEPS" != None ]]; then HYDRA_ARGS+=("trainer.total_training_steps=$TOTAL_STEPS"); fi
DATA_SEED=$(Y data_seed)
if [[ "$DATA_SEED" != None && -n "$DATA_SEED" ]]; then HYDRA_ARGS+=("data.seed=$DATA_SEED"); fi
for arg in "$@"; do
    case "$arg" in
        resume_dir=*|+resume_dir=*)
            RESUME_DIR="${arg#*=}"
            ;;
        swanlab_run_id=*|+swanlab_run_id=*)
            SWANLAB_RUN_ID_OVERRIDE="${arg#*=}"
            ;;
        train_dataset=*|+train_dataset=*)
            TRAIN_DATASET="${arg#*=}"
            ;;
        val_dataset=*|+val_dataset=*)
            VAL_DATASET="${arg#*=}"
            ;;
        data_dir=*|+data_dir=*)
            DATA_ROOT="${arg#*=}"
            ;;
        model_path=*|+model_path=*)
            MODEL_PATH="${arg#*=}"
            ;;
        train_files=*|+train_files=*)
            TRAIN_FILES="${arg#*=}"
            ;;
        val_files=*|+val_files=*)
            VAL_FILES="${arg#*=}"
            ;;
        val_max_samples=*|+val_max_samples=*)
            VAL_MAX_SAMPLES="${arg#*=}"
            ;;
        filter_overlong_prompts=*|+filter_overlong_prompts=*)
            FILTER_OVERLONG_PROMPTS="${arg#*=}"
            ;;
        truncation=*|+truncation=*)
            DATA_TRUNCATION="${arg#*=}"
            ;;
        gold_context_fallback_mode=*|+gold_context_fallback_mode=*)
            GOLD_CONTEXT_FALLBACK_MODE="${arg#*=}"
            ;;
        actor_param_offload=*|+actor_param_offload=*)
            ACTOR_PARAM_OFFLOAD="${arg#*=}"
            ;;
        actor_optimizer_offload=*|+actor_optimizer_offload=*)
            ACTOR_OPTIMIZER_OFFLOAD="${arg#*=}"
            ;;
        ref_param_offload=*|+ref_param_offload=*)
            REF_PARAM_OFFLOAD="${arg#*=}"
            ;;
        actor_ppo_micro_batch_size_per_gpu=*|+actor_ppo_micro_batch_size_per_gpu=*)
            ACTOR_PPO_MICRO_BS="${arg#*=}"
            ;;
        rollout_log_prob_micro_batch_size_per_gpu=*|+rollout_log_prob_micro_batch_size_per_gpu=*)
            ROLLOUT_LOGPROB_MICRO_BS="${arg#*=}"
            ;;
        ref_log_prob_micro_batch_size_per_gpu=*|+ref_log_prob_micro_batch_size_per_gpu=*)
            REF_LOGPROB_MICRO_BS="${arg#*=}"
            ;;
        experiment_name=*|+experiment_name=*)
            EXPERIMENT_NAME="${arg#*=}"
            ;;
        output_dir=*|+output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        privileged_text_mode=*|+privileged_text_mode=*)
            PRIVILEGED_TEXT_MODE="${arg#*=}"
            ;;
        math_privileged_text_mode=*|+math_privileged_text_mode=*)
            MATH_PRIVILEGED_TEXT_MODE="${arg#*=}"
            ;;
        val_enable_thinking=*|+val_enable_thinking=*)
            VAL_ENABLE_THINKING="${arg#*=}"
            ;;
        reward_manager_name=*|+reward_manager_name=*)
            REWARD_MANAGER_NAME="${arg#*=}"
            ;;
        use_custom_reward_function=*|+use_custom_reward_function=*)
            USE_CUSTOM_REWARD_FUNCTION="${arg#*=}"
            USE_CUSTOM_REWARD_FUNCTION_LOWER=$(echo "$USE_CUSTOM_REWARD_FUNCTION" | tr '[:upper:]' '[:lower:]')
            ;;
        custom_reward_function_path=*|+custom_reward_function_path=*)
            CUSTOM_REWARD_FUNCTION_PATH="${arg#*=}"
            ;;
        custom_reward_function_name=*|+custom_reward_function_name=*)
            CUSTOM_REWARD_FUNCTION_NAME="${arg#*=}"
            ;;
        data.train_files=*)
            TRAIN_FILES="${arg#*=}"
            ;;
        data.val_files=*)
            VAL_FILES="${arg#*=}"
            ;;
        *)
            HYDRA_ARGS+=("$arg")
            ;;
    esac
done

REWARD_ARGS=()
if [ -n "$REWARD_MANAGER_NAME" ] && [ "$REWARD_MANAGER_NAME" != "None" ]; then
    REWARD_ARGS+=(reward.reward_manager.name=${REWARD_MANAGER_NAME})
fi
if [ "$USE_CUSTOM_REWARD_FUNCTION_LOWER" = "true" ] \
    && [ -n "$CUSTOM_REWARD_FUNCTION_PATH" ] && [ "$CUSTOM_REWARD_FUNCTION_PATH" != "None" ] \
    && [ -n "$CUSTOM_REWARD_FUNCTION_NAME" ] && [ "$CUSTOM_REWARD_FUNCTION_NAME" != "None" ]; then
    REWARD_ARGS+=(
        "reward.custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH}"
        reward.custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME}
    )
fi

MICRO_BATCH_ARGS=()
if [ -n "$ROLLOUT_LOGPROB_MICRO_BS" ] && [ "$ROLLOUT_LOGPROB_MICRO_BS" != "None" ]; then
    MICRO_BATCH_ARGS+=(actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BS})
fi
if [ -n "$REF_LOGPROB_MICRO_BS" ] && [ "$REF_LOGPROB_MICRO_BS" != "None" ]; then
    MICRO_BATCH_ARGS+=(actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BS})
fi

PARQUET_ROOT="${DATA_ROOT%/}/verl"

normalize_dataset_spec() {
    local spec="$1"
    spec="$(echo "$spec" | tr -d '[:space:]')"
    spec="${spec#,}"
    spec="${spec%,}"
    while [[ "$spec" == *",,"* ]]; do
        spec="${spec//,,/,}"
    done
    printf '%s' "$spec"
}

dataset_spec_key() {
    local spec
    spec="$(normalize_dataset_spec "$1")"
    printf '%s' "${spec//,/+}"
}

normalize_privileged_text_mode() {
    local mode="$1"
    mode="$(echo "$mode" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$mode" in
        solution+answer|solution_and_answer)
            mode="solution_answer"
            ;;
        answer|answer_only)
            mode="answer_only"
            ;;
        gold_paragraphs|gold_context_only)
            mode="gold_context"
            ;;
    esac
    printf '%s' "$mode"
}

ensure_dataset_parquet() {
    local split="$1"
    local dataset_name="$2"
    local target_path="$3"

    if [ -f "$target_path" ]; then
        return 0
    fi
    echo "Missing ${split} parquet: ${target_path}" >&2
    echo "Run: python scripts/download_data.py --dataset ${dataset_name} --split ${split}" >&2
    exit 1
}

if [ -z "$TRAIN_FILES" ] || [ "$TRAIN_FILES" = "None" ]; then
    TRAIN_FILES="${PARQUET_ROOT}/${TRAIN_DATASET}/train.parquet"
    ensure_dataset_parquet train "$TRAIN_DATASET" "$TRAIN_FILES"
fi

if [ -z "$VAL_FILES" ] || [ "$VAL_FILES" = "None" ]; then
    VAL_DATASET="$(normalize_dataset_spec "$VAL_DATASET")"
    VAL_FILES="${PARQUET_ROOT}/$(dataset_spec_key "$VAL_DATASET")/val.parquet"
    ensure_dataset_parquet val "$VAL_DATASET" "$VAL_FILES"
fi

PRIVILEGED_TEXT_MODE="$(normalize_privileged_text_mode "$PRIVILEGED_TEXT_MODE")"

resolve_abs_path() {
    "$PYTHON_BIN" - "$1" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
}

discover_swanlab_run_id() {
    python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
experiment_name = sys.argv[2]

record_path = run_dir / "swanlab_run_id.txt"
if record_path.exists():
    run_id = record_path.read_text(encoding="utf-8").strip()
    if run_id:
        print(run_id)
        raise SystemExit(0)

swanlog_root = Path("swanlog")
if not swanlog_root.exists():
    raise SystemExit(0)

candidates = sorted(swanlog_root.glob("run-*/backup.swanlab"), key=lambda path: path.stat().st_mtime, reverse=True)
for backup_path in candidates:
    try:
        with backup_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(32):
                line = handle.readline()
                if not line:
                    break
                json_start = line.find("{")
                if json_start < 0:
                    continue
                payload = json.loads(line[json_start:])
                if payload.get("model_type") != "Experiment":
                    continue
                data = payload.get("data", {})
                if data.get("name") == experiment_name and data.get("id"):
                    print(data["id"])
                    raise SystemExit(0)
                break
    except Exception:
        continue
PY
}

TEE_ARGS=()
RESUME_ARGS=()

if [ -n "$RESUME_DIR" ] && [ "$RESUME_DIR" != "None" ]; then
    DIR="$(resolve_abs_path "${RESUME_DIR%/}")"
    if [ ! -d "$DIR" ]; then
        echo "Resume directory does not exist: $DIR" >&2
        exit 1
    fi
    if [ ! -f "$DIR/latest_checkpointed_iteration.txt" ]; then
        echo "Resume directory is missing latest_checkpointed_iteration.txt: $DIR" >&2
        exit 1
    fi

    RUN_TAG="$(basename "$DIR")"
    EXP="${EXPERIMENT_NAME}-${RUN_TAG}"
    TEE_ARGS=(-a)
    RESUME_ARGS+=(trainer.resume_mode=auto)
    export VERL_FILE_LOGGER_APPEND=1

    if [ "$USE_SWANLAB_LOWER" = "true" ]; then
        export SWANLAB_RUN_ID_RECORD_FILE="${DIR}/swanlab_run_id.txt"
        if [ -n "$SWANLAB_RUN_ID_OVERRIDE" ] && [ "$SWANLAB_RUN_ID_OVERRIDE" != "None" ]; then
            export SWANLAB_RUN_ID="$SWANLAB_RUN_ID_OVERRIDE"
        else
            SWANLAB_RUN_ID_FOUND="$(discover_swanlab_run_id "$DIR" "$EXP")"
            if [ -z "$SWANLAB_RUN_ID_FOUND" ]; then
                echo "Failed to discover SwanLab run id for ${EXP}. Pass swanlab_run_id=<run_id>." >&2
                exit 1
            fi
            export SWANLAB_RUN_ID="$SWANLAB_RUN_ID_FOUND"
            printf '%s\n' "$SWANLAB_RUN_ID_FOUND" > "${DIR}/swanlab_run_id.txt"
        fi
        export SWANLAB_RESUME=must
    else
        unset SWANLAB_RUN_ID
        unset SWANLAB_RESUME
        unset SWANLAB_RUN_ID_RECORD_FILE
    fi
else
    TS=$(date +%Y%m%d.%H%M%S)
    EXP="${EXPERIMENT_NAME}-${TS}"
    DIR="${OUTPUT_DIR%/}/${TS}"
    mkdir -p "$DIR"
    cp "$CONFIG" "$DIR/config.yaml"
    unset VERL_FILE_LOGGER_APPEND
    if [ "$USE_SWANLAB_LOWER" = "true" ]; then
        unset SWANLAB_RUN_ID
        unset SWANLAB_RESUME
        export SWANLAB_RUN_ID_RECORD_FILE="${DIR}/swanlab_run_id.txt"
    else
        unset SWANLAB_RUN_ID
        unset SWANLAB_RESUME
        unset SWANLAB_RUN_ID_RECORD_FILE
    fi
fi

mkdir -p "$DIR"
if [ "$(resolve_abs_path "$CONFIG")" != "$(resolve_abs_path "$DIR/config.yaml")" ]; then
    cp "$CONFIG" "$DIR/config.yaml"
fi
export TENSORBOARD_DIR="${DIR}/tensorboard_log/"
export VERL_FILE_LOGGER_PATH="${DIR}/metrics.jsonl"

# --- Build algorithm-specific overrides from YAML ---
ALGO=""
for key in beta jsd_token_clip teacher_mode \
           full_logit_distill distill_add_tail distill_coef distill_wrong_only distill_certified_only \
           distill_key_ablation_gate distill_key_ablation_gate_mode \
           distill_key_ablation_tau \
           distill_key_ablation_saturation distill_key_ablation_field_tag \
           distill_key_ablation_gate_token_denominator \
           distill_fwd_mode distill_fwd_coef \
           distill_fwd_teacher_topp distill_fwd_hard_threshold \
           distill_coef_correct \
           alpha top_k_distill is_clip ema_decay \
           epsilon epsilon_w tau evidence_gate_tau ce_gate_tau lam lam_decay_steps teacher_sync_interval teacher_sync_unit \
           evidence_diagnostics teacher_gate_mode teacher_gate_tau \
           rlsd_delta_source rlsd_onesided_clamp rlsd_penalty_only rlsd_boost_prob_ceiling rlsd_skip_math \
           rlsd_shuffle_placebo \
           rlcsd_tau rlcsd_beta rlcsd_lam rlcsd_delta rlcsd_eta \
           rlcsd_residual_clip_low rlcsd_residual_clip_high rlcsd_k_max \
           opsd_ectr_alpha opsd_ectr_log_w_floor opsd_ectr_cw_tau \
           rlsd_ectr_lam \
           srpo_beta; do
    val=$(Y $key "")
    [ -n "$val" ] && [ "$val" != "None" ] && ALGO="$ALGO +actor_rollout_ref.actor.policy_loss.${key}=${val}"
done
if [ "$METHOD" != "grpo" ] && [ "$METHOD" != "opd" ]; then
    # tailfmt contract (2026-08-11): preset NAME for the tail reminder that
    # teacher/evidence views must replicate (resolved in opsd_format.py)
    VIEW_PROMPT_TAIL_PRESET=$(Y view_prompt_tail_preset none)
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.view_prompt_tail_preset=${VIEW_PROMPT_TAIL_PRESET}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.privileged_text_mode=${PRIVILEGED_TEXT_MODE}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.math_privileged_text_mode=${MATH_PRIVILEGED_TEXT_MODE}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.gold_context_fallback_mode=${GOLD_CONTEXT_FALLBACK_MODE}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.teacher_enable_thinking=${TEACHER_ENABLE_THINKING}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.thinking_system_prompt=${THINKING_SYSTEM_PROMPT}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.teacher_view=$(Y teacher_view gold_only)"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.bullets_generator_tag=$(Y bullets_generator_tag qwen25_72b)"
fi

# --- OPD-specific distillation overrides ---
DISTILL_ARGS=()
if [ "$METHOD" = "opd" ]; then
    DISTILL_ARGS=(
        distillation.enabled=True
        distillation.num_workers=${DISTILL_NUM_WORKERS}
        distillation.teacher_model.enable_resource_pool=${TEACHER_ENABLE_RESOURCE_POOL}
        distillation.teacher_model.n_gpus_per_node=${TEACHER_N_GPUS_PER_NODE}
        distillation.teacher_model.nnodes=1
        distillation.teacher_model.model_path=${TEACHER_MODEL_PATH}
        distillation.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP_SIZE}
        distillation.teacher_model.inference.name=vllm
        distillation.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM}
        distillation.teacher_model.inference.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
        distillation.teacher_model.inference.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}
        distillation.distillation_loss.loss_mode=${DISTILL_LOSS_MODE}
        distillation.distillation_loss.topk=${DISTILL_TOPK}
        distillation.distillation_loss.use_task_rewards=${DISTILL_USE_TASK_REWARDS}
        distillation.distillation_loss.use_policy_gradient=${DISTILL_USE_PG}
        distillation.distillation_loss.distillation_loss_coef=${DISTILL_LOSS_COEF}
        distillation.distillation_loss.loss_max_clamp=${DISTILL_LOSS_MAX_CLAMP}
        distillation.distillation_loss.log_prob_min_clamp=${DISTILL_LOG_PROB_MIN_CLAMP}
    )
fi

ROLLOUT_CORR=""
for key in rollout_is rollout_is_threshold rollout_is_batch_normalize rollout_rs rollout_rs_threshold; do
    val=$(Y $key "")
    [ -n "$val" ] && [ "$val" != "None" ] && ROLLOUT_CORR="$ROLLOUT_CORR algorithm.rollout_correction.${key}=${val}"
done

CLIP_ARGS=()
if [ -n "$CLIP_RATIO_LOW" ] && [ "$CLIP_RATIO_LOW" != "None" ]; then
    CLIP_ARGS+=("actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}")
fi
if [ -n "$CLIP_RATIO_HIGH" ] && [ "$CLIP_RATIO_HIGH" != "None" ]; then
    CLIP_ARGS+=("actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}")
fi

MODEL_OVERRIDE=""
ROLLOUT_MODEL_OVERRIDE_ARGS=()
if [ -n "$OVERRIDE_MAX_POSITION_EMBEDDINGS" ] && [ "$OVERRIDE_MAX_POSITION_EMBEDDINGS" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.max_position_embeddings=${OVERRIDE_MAX_POSITION_EMBEDDINGS}"
    if [ "$OVERRIDE_MAX_POSITION_EMBEDDINGS" -lt "$ROLLOUT_MAX_MODEL_LEN" ]; then
        echo "override_max_position_embeddings=${OVERRIDE_MAX_POSITION_EMBEDDINGS} is smaller than required max_model_len=${ROLLOUT_MAX_MODEL_LEN}" >&2
        exit 1
    fi
fi
if [ -n "$ROPE_SCALING_TYPE" ] && [ "$ROPE_SCALING_TYPE" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.type=${ROPE_SCALING_TYPE}"
fi
if [ -n "$ROPE_SCALING_FACTOR" ] && [ "$ROPE_SCALING_FACTOR" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.factor=${ROPE_SCALING_FACTOR}"
fi
if [ -n "$ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS" ] && [ "$ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=${ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS}"
fi
if [ -n "$MODEL_OVERRIDE" ]; then
    ROLLOUT_HF_OVERRIDES=$(
        "$PYTHON_BIN" - \
            "$OVERRIDE_MAX_POSITION_EMBEDDINGS" \
            "$ROPE_SCALING_TYPE" \
            "$ROPE_SCALING_FACTOR" \
            "$ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS" <<'PY'
import json
import sys

max_positions, rope_type, rope_factor, rope_original = sys.argv[1:]
overrides = {}
if max_positions not in ("", "None"):
    overrides["max_position_embeddings"] = int(max_positions)
rope_values = (rope_type, rope_factor, rope_original)
if any(value not in ("", "None") for value in rope_values):
    if not all(value not in ("", "None") for value in rope_values):
        raise SystemExit(
            "rope_scaling_type, rope_scaling_factor, and "
            "rope_scaling_original_max_position_embeddings must be set together"
        )
    overrides["rope_scaling"] = {
        "type": rope_type,
        "factor": float(rope_factor),
        "original_max_position_embeddings": int(rope_original),
    }
print(json.dumps(overrides, separators=(",", ":")))
PY
    )
    # vLLM reloads config.json from model.path, so explicitly mirror the actor's
    # in-memory Hugging Face overrides into the rollout engine.
    ROLLOUT_MODEL_OVERRIDE_ARGS+=(
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='${ROLLOUT_HF_OVERRIDES}'"
    )
fi

if [ -n "$KL_LOSS_COEF" ] && [ "$KL_LOSS_COEF" != "None" ]; then
    EFFECTIVE_KL_LOSS_COEF="$KL_LOSS_COEF"
else
    case "$METHOD" in
        grpo) EFFECTIVE_KL_LOSS_COEF="0.001" ;;
        *) EFFECTIVE_KL_LOSS_COEF="0" ;;
    esac
fi
case "$METHOD" in
    grpo|opd) EFFECTIVE_POLICY_LOSS_MODE="vanilla" ;;
    *)        EFFECTIVE_POLICY_LOSS_MODE="$METHOD" ;;
esac

# use_kl_loss: default true (ref forward kept). Setting false is only allowed
# for methods without teacher plumbing (grpo/opd); distill teacher log probs
# are computed through the ref/kl code path even when kl_loss_coef=0.
EFFECTIVE_USE_KL_LOSS=True
if [ -n "$USE_KL_LOSS" ] && [ "$USE_KL_LOSS" != "None" ]; then
    USE_KL_LOSS_LOWER=$(echo "$USE_KL_LOSS" | tr '[:upper:]' '[:lower:]')
    if [ "$USE_KL_LOSS_LOWER" = "false" ]; then
        case "$METHOD" in
            grpo|opd) EFFECTIVE_USE_KL_LOSS=False ;;
            *)
                echo "use_kl_loss=false is not supported for method='${METHOD}' (teacher rides on ref/kl path)" >&2
                exit 1
                ;;
        esac
    fi
fi

RRS_ARGS=()
RRS_LOWER=$(echo "${REWARD_REJECTION_SAMPLING:-}" | tr '[:upper:]' '[:lower:]')
if [ "$RRS_LOWER" = "true" ]; then
    RRS_ARGS+=("trainer.reward_rejection_sampling=True")
fi

FILTER_GROUPS_ARGS=()
FILTER_GROUPS_ENABLE_LOWER=$(echo "${FILTER_GROUPS_ENABLE:-}" | tr '[:upper:]' '[:lower:]')
if [ "$FILTER_GROUPS_ENABLE_LOWER" = "true" ]; then
    if [ "$RRS_LOWER" = "true" ]; then
        echo "reward_rejection_sampling and filter_groups_enable are mutually exclusive" >&2
        exit 1
    fi
    FILTER_GROUPS_ARGS+=(
        "algorithm.filter_groups.enable=True"
        "algorithm.filter_groups.max_num_gen_batches=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES}"
    )
    if [ -n "$GEN_BATCH_SIZE" ] && [ "$GEN_BATCH_SIZE" != "None" ]; then
        FILTER_GROUPS_ARGS+=("+data.gen_batch_size=${GEN_BATCH_SIZE}")
    fi
fi

# --- Use custom main for privileged-teacher distillation, standard for grpo / opd ---
MAIN=src.self_distill_main
case "$METHOD" in
    grpo|opd) MAIN=verl.trainer.main_ppo ;;
esac

WORKER_IMPL=$(Y use_legacy_worker_impl "")
if [ -z "$WORKER_IMPL" ] || [ "$WORKER_IMPL" = "None" ]; then
    case "$METHOD" in
        grpo|opd) WORKER_IMPL=disable ;;
        *)        WORKER_IMPL=enable ;;
    esac
fi

"$PYTHON_BIN" -m $MAIN \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "${RESUME_ARGS[@]}" \
    trainer.val_before_train=${VAL_BEFORE} \
    trainer.use_legacy_worker_impl=${WORKER_IMPL} \
    trainer.critic_warmup=0 \
    trainer.logger=${LOGGER} \
    trainer.project_name=${PROJECT} \
    trainer.experiment_name=${EXP} \
    "trainer.default_local_dir=${DIR}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_STEPS} \
    trainer.test_freq=${EVAL_STEPS} \
    trainer.total_epochs=${NUM_EPOCHS} \
    "data.train_files=${TRAIN_FILES}" \
    "data.val_files=${VAL_FILES}" \
    data.train_batch_size=${ROLLOUT_BATCH} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.train_max_samples=${MAX_TRAIN} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESP} \
    +data.apply_chat_template_kwargs.enable_thinking=${STUDENT_ENABLE_THINKING} \
    +data.apply_chat_template_kwargs.force_think_prefix=${FORCE_THINK_PREFIX} \
    +data.val_apply_chat_template_kwargs.enable_thinking=${VAL_ENABLE_THINKING} \
    +data.val_apply_chat_template_kwargs.force_think_prefix=${FORCE_THINK_PREFIX} \
    +data.thinking_system_prompt=${THINKING_SYSTEM_PROMPT} \
    data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS} \
    data.truncation=${DATA_TRUNCATION} \
    data.shuffle=${DATA_SHUFFLE} \
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT} \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "${LORA_ARGS[@]}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP} \
    actor_rollout_ref.actor.grad_clip=${MAX_GRAD_NORM} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BS} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SP} \
    actor_rollout_ref.actor.use_kl_loss=${EFFECTIVE_USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${EFFECTIVE_KL_LOSS_COEF} \
    "${FILTER_GROUPS_ARGS[@]}" \
    "${RRS_ARGS[@]}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${EFFECTIVE_POLICY_LOSS_MODE} \
    actor_rollout_ref.rollout.temperature=${TEMP} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_SEQ_LEN} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM} \
    actor_rollout_ref.rollout.n=${GROUP_SIZE} \
    actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT} \
    actor_rollout_ref.rollout.layered_summon=${ROLLOUT_LAYERED_SUMMON} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB} \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS} \
    "${MICRO_BATCH_ARGS[@]}" \
    +actor_rollout_ref.rollout.custom.val_response_length=${VAL_MAX_RESP} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_SEQ_LEN} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
    actor_rollout_ref.ref.strategy=${ACTOR_STRATEGY} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${REF_MODEL_DTYPE} \
    "${REWARD_ARGS[@]}" \
    ${MODEL_OVERRIDE} \
    "${ROLLOUT_MODEL_OVERRIDE_ARGS[@]}" \
    ${ALGO} \
    ${ROLLOUT_CORR} \
    "${CLIP_ARGS[@]}" \
    "${DISTILL_ARGS[@]}" \
    "${HYDRA_ARGS[@]}" \
    2>&1 | tee "${TEE_ARGS[@]}" "${DIR}/train_log.txt"
