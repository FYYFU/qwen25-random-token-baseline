"""Verify the experiment inputs and write the absolute-path veRL configuration."""
import argparse
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHTS_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
# The historical launcher concatenates these splits before constructing its sampler.
TRAIN_SPLITS = (
    "hotpotqa_qwen_0_2500", "musique_qwen_0_2500", "2wikipedia_qwen_0_2500",
    "hotpotqa_distractor_2500_5000", "musique_distractor_2500_5000",
    "2wikipedia_distractor_2500_5000", "mathqa_choice", "dapo",
    "ruler_multikey", "ruler_multivalue",
)
MODEL_BLOBS = {
    "config.json": "f81ead14ab072d65a07817f83a3ee0e5a1890d10",
    "generation_config.json": "dfc11073787daf1b0f9c0f1499487ab5f4c93738",
    "merges.txt": "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
    "tokenizer.json": "443909a61d429dff23010e5bddd28ff530edda00",
    "tokenizer_config.json": "07bfe0640cb5a0037f9322287fbfc682806cf672",
    "vocab.json": "4783fe10ac3adce15ac8f358ef5462739852c569",
}

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def verify_data(profile):
    folder = ROOT / "data" / profile
    if profile == "full":
        train = [folder / split / "train.parquet" for split in TRAIN_SPLITS]
        val = folder / "mathqa_choice" / "valid.parquet"
    else:
        train, val = [folder / "train.parquet"], folder / "valid.parquet"
    sums = {}
    for line in (folder / "checksums.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name] = digest
    required = {str(p.relative_to(folder)) for p in [*train, val]}
    if profile == "full":
        required.add("manifest.json")
    if set(sums) != required:
        raise SystemExit(f"Unexpected checksum manifest for {profile}")
    for name, expected in sums.items():
        path = folder / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"Missing or incorrect file: {path}. For full data, run git lfs pull.")
    return train, val

def model_dir():
    if local := os.getenv("MODEL_PATH"):
        path = Path(local).expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(MODEL_ID, revision=REVISION, cache_dir=ROOT / "models" / "hub",
                                     allow_patterns=[*MODEL_BLOBS, "model.safetensors"]))
    if sha256(path / "model.safetensors") != WEIGHTS_SHA256:
        raise SystemExit(f"Base weights differ from {MODEL_ID}@{REVISION}")
    for name, expected in MODEL_BLOBS.items():
        blob = (path / name).read_bytes()
        actual = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
        if actual != expected:
            raise SystemExit(f"Unexpected revision of {name}")
    return path

def write_config(profile, train, val, model, target):
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/random.yaml").read_text())
    if profile == "pilot":
        cfg.update(per_device_batch_size=2, ppo_mini_batch_size=8, total_training_steps=1,
                   data_seed=42, val_max_samples=-1, save_steps=-1, n_gpus_per_node=4,
                   vllm_tensor_parallel_size=1, vllm_gpu_memory_utilization=0.5,
                   actor_ppo_micro_batch_size_per_gpu=1, update_weights_bucket_megabytes=1024)
    target = target.resolve()
    cfg.update(model_path=str(model), train_files=[str(p) for p in train],
               val_files=[str(val)], custom_reward_function_path=str(ROOT / "training/loongrl_reward_repro.py"),
               experiment_name=f"qwen25_random_{profile}", output_dir=str(target.parent))
    target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Saved {target}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("pilot", "full"))
    parser.add_argument("--output-config", type=Path,
                        help="also prepare the pinned model and write the launch config into a run directory")
    args = parser.parse_args()
    train, val = verify_data(args.profile)
    print(f"Verified {len(train)} training parquet(s) and {val.name} for {args.profile}.")
    if args.output_config is None:
        return
    write_config(args.profile, train, val, model_dir(), args.output_config)

if __name__ == "__main__":
    main()
