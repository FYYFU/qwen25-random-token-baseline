#!/usr/bin/env bash
# Run in the staged repository on a single 8-GPU OBX node.
set -euo pipefail
arm=${1:?Choose random, kl-topk, or certified-dense}
case "$arm" in random|kl-topk|certified-dense) ;; *) echo "Unknown experiment: $arm" >&2; exit 2 ;; esac
: "${TRAIN_PYTHON:?Absolute Python path for the training environment}"
: "${EVAL_PYTHON:?Absolute Python path for the evaluation environment}"
: "${OUTPUT_S3:?S3 prefix for checkpoints and scores}"
[[ "$OUTPUT_S3" == s3://* && -x "$TRAIN_PYTHON" && -x "$EVAL_PYTHON" ]]
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
log=$(mktemp)
trap 'rm -f "$log"' EXIT
PYTHON_BIN="$TRAIN_PYTHON" bash run.sh full "$arm" 2>&1 | tee "$log"
prefix="Qwen2.5 $arm (full) logs and output: "
run=$(awk -v prefix="$prefix" 'index($0, prefix) == 1 {print substr($0,length(prefix)+1); exit}' "$log")
[[ -n "$run" && -d "$run" ]]
actors=("$run"/*/global_step_41/actor)
[[ ${#actors[@]} -eq 1 && -d "${actors[0]}" ]]
model="$run/merged-step41"
results="$run/eval-step41"
dest="${OUTPUT_S3%/}/$(basename "$run")"
aws s3 cp "$run/trial.yaml" "$dest/trial.yaml"
aws s3 sync "${actors[0]}" "$dest/global_step_41/actor/" --only-show-errors
"$TRAIN_PYTHON" eval/run.py merge "${actors[0]}" "$model"
aws s3 sync "$model" "$dest/merged-step41/" --only-show-errors

# Preserve completed generation and scoring outputs if a later phase fails.
save_eval() {
    status=$?
    if [[ -d "$results" ]]; then
        aws s3 sync "$results" "$dest/eval-step41/" --only-show-errors || status=1
    fi
    rm -f "$log"
    exit "$status"
}
trap save_eval EXIT
"$EVAL_PYTHON" eval/run.py generate --longbench-only "$model" "$results"
"$EVAL_PYTHON" eval/run.py judge "$results"
"$EVAL_PYTHON" eval/run.py report --longbench-only "$results"
echo "LongBench QA scores: $dest/eval-step41/scores.json"
