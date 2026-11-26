#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root="$PWD"
case "${1:-}" in
    pilot|full) profile=$1 ;;
    *) echo "Usage: ./run.sh {pilot|full}" >&2; exit 2 ;;
esac
[[ $# == 1 ]] || { echo "Expected a single profile argument." >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ $profile == full ]]; then
    expected_gpus=8
    default_devices=0,1,2,3,4,5,6,7
else
    expected_gpus=4
    default_devices=0,1,2,3
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-$default_devices}"
IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
    [[ ${#devices[@]} -ne $expected_gpus || $(printf '%s\n' "${devices[@]}" | sort -u | wc -l) -ne $expected_gpus ]]; then
    echo "$profile requires $expected_gpus distinct physical GPU indices." >&2
    exit 2
fi
if [[ -z "${PYTHON_BIN:-}" && -x .venv/bin/python ]]; then PYTHON_BIN="$root/.venv/bin/python"; fi
export PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}")"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$root/RLCSD/third_party/verl:$root/RLCSD${PYTHONPATH:+:$PYTHONPATH}"
export RAY_ADDRESS=local RAY_DEDUP_LOGS=0 TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}" NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export REWARD_CALC_TYPE=pure_exact_match RULER_REWARD_CALC_TYPE=format_exact_match
mkdir -p "$root/runs/$profile"
run="$(mktemp -d "$root/runs/$profile/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
echo "Qwen2.5 Random ($profile) logs and output: $run"
"$PYTHON_BIN" scripts/prepare.py "$profile" --output-config "$run/trial.yaml"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CFOPSDL_TELEMETRY_DIR="$run"
CUDA_VISIBLE_DEVICES="${devices[0]}" "$PYTHON_BIN" scripts/check_random_gate.py --qwen2
date +%s > "$run/start_unix"
"$PYTHON_BIN" training/sample_trial_gpu.py "$run" >"$run/monitor.log" 2>&1 &
monitor=$!
finish() {
    result=$?
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    date +%s > "$run/end_unix"
    echo "$result" > "$run/exit_code"
}
trap finish EXIT
if [[ $profile == pilot ]]; then
    options=(data.dataloader_num_workers=0 ray_kwargs.ray_init.num_cpus=20
        "+ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_BYTES:-4294967296}"
        actor_rollout_ref.rollout.agent.num_workers=8 actor_rollout_ref.rollout.enforce_eager=True
        actor_rollout_ref.rollout.max_num_seqs=32 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
        actor_rollout_ref.actor.use_torch_compile=False +actor_rollout_ref.actor.response_only_lm_head=True)
else
    options=(data.dataloader_num_workers=8 actor_rollout_ref.rollout.agent.num_workers=8
        actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.max_num_seqs=1024
        actor_rollout_ref.actor.entropy_from_logits_with_chunking=False actor_rollout_ref.actor.use_torch_compile=True)
    if [[ -n "${RAY_OBJECT_STORE_BYTES:-}" ]]; then
        options+=("+ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_BYTES")
    fi
fi
bash RLCSD/scripts/_run_verl.sh "$run/trial.yaml" "${options[@]}" 2>&1 | tee "$run/console.log"
