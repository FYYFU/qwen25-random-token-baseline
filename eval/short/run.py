"""Run the six historical short-context tasks on one merged checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run import digest, lock, model_files
from score import DATA
from validate_short_general_result import latest_result, task_block, validate_mmlu, validate_non_mmlu


def run(model, root):
    weights = model_files(model)
    identity = root / "model.json"
    if identity.is_file() and json.loads(identity.read_text()) != weights:
        raise ValueError("Different model. Use a new output directory.")
    if not identity.exists():
        if (root / "mmlu").exists() or (root / "non_mmlu").exists():
            raise ValueError("Unidentified existing scores. Use a new output directory.")
        tmp = identity.with_suffix(".tmp")
        tmp.write_text(json.dumps(weights, indent=2) + "\n")
        tmp.replace(identity)
    archive = DATA / "short/datasets.tar.gz"
    entries = (line.split() for line in (DATA / "checksums.sha256").read_text().splitlines())
    hashes = {name: expected for expected, name in entries}
    if digest(archive) != hashes["short/datasets.tar.gz"]:
        raise ValueError("Incorrect short-context data. Run git lfs pull.")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not gpu.isdigit():
        raise ValueError("Set CUDA_VISIBLE_DEVICES to a numeric GPU index.")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HOME=str(root / "hf_home"), HF_DATASETS_CACHE=str(root / "hf_home/datasets"),
               TOKENIZERS_PARALLELISM="false", NCCL_NET_PLUGIN="none", NCCL_NVLS_ENABLE="0")
    cache = Path(env["HF_DATASETS_CACHE"])
    if not cache.is_dir():
        cache.parent.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=cache.parent) as tmp:
            staged = Path(tmp) / "datasets"
            staged.mkdir()
            with tarfile.open(archive) as tar:
                for member in tar:
                    if not member.isfile() or not (staged / member.name).resolve().is_relative_to(staged):
                        raise ValueError(f"Unexpected cache member: {member.name}")
                    tar.extract(member, staged)
            staged.rename(cache)
    base = [sys.executable, "-m", "lm_eval", "run", "--model", "vllm", "--model_args",
            f"pretrained={model}", "dtype=bfloat16", "tensor_parallel_size=1", "max_model_len=12288",
            "gpu_memory_utilization=0.90", "seed=0", "--apply_chat_template",
            "--batch_size", "auto", "--seed", "0", "--log_samples"]
    phases = {"mmlu": (["--tasks", "mmlu", "--num_fewshot", "5"], validate_mmlu),
              "non_mmlu": (["--tasks", "short_general_non_mmlu", "--include_path", str(HERE / "tasks")],
                           validate_non_mmlu)}
    scores = {}
    for name, (args, validate) in phases.items():
        dest = root / name
        if not dest.exists():
            with TemporaryDirectory(dir=root) as tmp:
                stage = Path(tmp) / "results"
                subprocess.run([*base, *args, "--output_path", str(stage)], env=env, check=True)
                validate(json.loads(latest_result(stage).read_text()))
                stage.rename(dest)
        payload = json.loads(latest_result(dest).read_text())
        validate(payload)
        metrics = {"mmlu": ["acc"]} if name == "mmlu" else {
            "arc_challenge": ["acc_norm"], "hellaswag": ["acc_norm"], "winogrande": ["acc"],
            "minerva_math": ["math_verify"], "ifeval": ["prompt_level_strict_acc", "inst_level_strict_acc"]}
        for task, names in metrics.items():
            scores[task] = {m: task_block(payload, task)[m + ",none"] for m in names}
    report = json.dumps(scores, indent=2)
    (root / "scores.json").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    with lock(args.results.resolve()):
        run(args.model.resolve(), args.results.resolve())
