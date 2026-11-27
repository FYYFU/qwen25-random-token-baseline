# Train and evaluate on OBX

Submit one single-node 8×H200 job per experiment. Choose `random`, `kl-topk`, or `certified-dense`. Each job trains Qwen2.5-1.5B for 41 steps, merges the final actor, and evaluates it on five LongBench QA tasks (2WikiMQA, HotpotQA, MuSiQue, NarrativeQA, and Qasper). All three jobs can use the same code and data bundle.

## Prepare the image and S3 bundle

Use a `p5en.48xlarge` image with the AWS CLI, CUDA toolkit, and both Python environments. Follow the [training setup](../README.md) for Python 3.12 and the [LongBench setup](../eval/README.md) for Python 3.11. Allow disk space for five training checkpoints, the merged model, and evaluation outputs. The job needs access to download the pinned Qwen base model or a local copy supplied via `MODEL_PATH`.

From a fresh clone on a machine with Git LFS, Bash, and the AWS CLI, stage the code and data once. Set `CODE_S3_URI` to a private S3 object shared by the three jobs. Using the commit in the object name makes it easy to tell which version they ran.

```bash
set -euo pipefail
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/FYYFU/qwen25-random-token-baseline.git
cd qwen25-random-token-baseline
git lfs install --local
git lfs pull --include='data/full/**,data/eval/longbench/**'
(cd data/full && sha256sum -c checksums.sha256)
(cd data/eval && awk '$2 ~ /^longbench\//' checksums.sha256 | sha256sum -c -)

CODE_S3_URI="s3://YOUR-BUCKET/rebuttal/code-$(git rev-parse --short HEAD).tar.gz"
bundle=$(mktemp -d)
git ls-files -z -- . ':!data/eval/longreason/**' ':!data/eval/short/**' | COPYFILE_DISABLE=1 tar -czf "$bundle/code.tar.gz" --null -T -
aws s3 cp "$bundle/code.tar.gz" "$CODE_S3_URI"
echo "$CODE_S3_URI"
```

The archive includes the full training mix, the five LongBench QA input files, and the [training and evaluation script](train-eval.sh).

## Submit your experiment

1. From the repository root, run `cp obx/job.example.json job.json`. In your copy of the [job template](job.example.json), set `JobName` and `jobDefinitionName` to a name that identifies your experiment, and set `EXPERIMENT` to its exact name.
2. Fill in your OBX initiative, role, region, container image, the `CODE_S3_URI` printed above, a private `OUTPUT_S3` prefix, and the absolute paths to the training and evaluation Python executables in the image. The job role needs to read the bundle and list and write to the results prefix. LongBench judging also needs Bedrock `InvokeModel` access to `openai.gpt-oss-120b-1:0` in `ap-south-1`. If your team uses a named AWS profile, reuse the profile setup from your working OBX job.
3. Set `OBX_SUBMITTER` to the local path to your team's `submit_obx.py`. Preview and submit the job from the directory containing your edited `job.json`.

```bash
python "$OBX_SUBMITTER" job.json --dry-run
python "$OBX_SUBMITTER" job.json
```

## Share the results

Send the job ID when training starts. After the job finishes, share the S3 link to `eval-step41/scores.json`. It reports all five LongBench QA scores and their unweighted average on a 0–100 scale.

The job uploads the step-41 actor and merged model before evaluation. If a later evaluation stage fails, the entry script attempts to upload completed records on exit. To finish scoring those records, follow the [evaluation guide](../eval/README.md).
