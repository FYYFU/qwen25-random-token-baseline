"""Merge a step-41 actor, generate the two long-context suites, judge and report."""
import argparse
import asyncio
from contextlib import contextmanager, suppress
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import URLError
from urllib.request import urlopen

import score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# (context window, maximum server sequences, YaRN factor)
UNITS = {"longbench": (41984, 32, 2), "8k": (24576, 32, 1), "16k": (32768, 32, 1),
         "32k": (41984, 32, 2), "64k": (73728, 16, 3)}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs():
    for line in (score.DATA / "checksums.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if digest(score.DATA / name) != expected:
            raise ValueError(f"Incorrect or missing LFS input {name}. Run git lfs pull.")


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".eval.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another evaluation command is using {directory}")
        yield


def model_files(model):
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError(f"Expected a merged Hugging Face model in {model}")
    return {p.name: digest(p) for p in sorted(model.iterdir())
            if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".model")}


def merge(actor, target):
    if not list(actor.glob("model_world_size_*_rank_*.pt")):
        raise ValueError(f"No FSDP actor shards in {actor}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(ROOT / "RLCSD/third_party/verl"),
                                          str(ROOT / "RLCSD"), os.environ.get("PYTHONPATH", "")]))
    with TemporaryDirectory(dir=target.parent) as tmp:
        merged = Path(tmp) / "model"
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", str(actor), "--target_dir", str(merged)], env=env, check=True)
        model_files(merged)
        merged.rename(target)


def commands(unit, model, output, port):
    window, seqs, factor = UNITS[unit]
    qa = unit == "longbench"
    name = "kc" if qa else "m"
    hf = {} if factor == 1 else {"max_position_embeddings": 32768 * factor, "rope_scaling": {
        "rope_type": "yarn", "factor": float(factor), "original_max_position_embeddings": 32768}}
    server = [str(Path(sys.executable).parent / "vllm"), "serve", str(model),
              "--host", "127.0.0.1", "--port", str(port), "--served-model-name", name,
              "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.92",
              "--max-model-len", str(window), "--max-num-seqs", str(seqs), "--hf-overrides", json.dumps(hf)]
    if qa:
        generator = [sys.executable, str(HERE / "longbench/eval_longbench_boxed_kc.py"),
                     "--contract", "think", "--envelope", "passages", "--v1-dir", str(score.DATA / "longbench")]
    else:
        generator = [sys.executable, str(HERE / "longreason/eval_longreason.py"), "--tier", unit,
                     "--contract", "think_mcq", "--data-dir", str(score.DATA / "longreason"),
                     "--save-text", "--gzip-records", "--slack", "1024"]
    generator += ["--model-path", str(model), "--served-name", name, "--out-dir", str(output),
                  "--api-base", f"http://127.0.0.1:{port}/v1", "--max-model-len", str(window),
                  "--max-tokens", "8192", "--temperature", "1", "--top-p", "1", "--n", "5",
                  "--seed", "0", "--workers", "16"]
    return server, generator


def ready(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as resp:
            return bool(json.load(resp)["data"])
    except (URLError, OSError):
        return False


async def stop(proc):
    if proc is None:
        return
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=20)
    except asyncio.TimeoutError:
        pass
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()


async def run_unit(unit, gpu, model, root):
    qa = unit == "longbench"
    dest = root / ("longbench/model" if qa else f"longreason/{unit}")
    def validate(directory):
        if qa:
            score.longbench(directory, judged=False)
        else:
            score.longreason(directory, unit)
    if dest.is_dir():
        validate(dest)
        print(f"Using complete {unit} records", flush=True)
        return
    dest.parent.mkdir(exist_ok=True)
    logfile = root / f"{unit}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, NCCL_NET_PLUGIN="none", NCCL_NVLS_ENABLE="0")
    with TemporaryDirectory(dir=root) as tmp, logfile.open("a") as log:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        staged = Path(tmp) / "records"
        server, generator = commands(unit, model, staged, port)
        async def launch(cmd):
            return await asyncio.create_subprocess_exec(*cmd, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Starting {unit} on GPU {gpu}. Log: {logfile}", flush=True)
        service = await launch(server)
        client = None
        try:
            deadline = time.monotonic() + 1200
            while not await asyncio.to_thread(ready, port):
                if service.returncode is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"{unit} server failed to start. See {logfile}")
                await asyncio.sleep(2)
            client = await launch(generator)
            if await client.wait():
                raise RuntimeError(f"{unit} generation failed. See {logfile}")
        finally:
            await stop(client)
            await stop(service)
        validate(staged)
        staged.rename(dest)
    print(f"Completed {unit}", flush=True)


async def generate(model, root):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if any(not d.isdigit() for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must list distinct numeric GPU indices")
    verify_inputs()
    fingerprint = {"model": model_files(model), "inputs": digest(score.DATA / "checksums.sha256")}
    manifest = root / "generation_inputs.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != fingerprint:
            raise ValueError("The weights, tokenizer or data differ. Choose a new results directory.")
    else:
        if (root / "longbench").exists() or (root / "longreason").exists():
            raise ValueError("Unidentified prior generations. Choose a new results directory.")
        tmp = manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(fingerprint, indent=2) + "\n")
        tmp.replace(manifest)
    units = list(UNITS)
    async def worker(i, gpu):
        for unit in units[i::len(devices)]:
            await run_unit(unit, gpu, model, root)
    async with asyncio.TaskGroup() as tasks:
        for i, gpu in enumerate(devices[:len(units)]):
            tasks.create_task(worker(i, gpu))


def judge(root):
    folder = root / "longbench/model"
    score.longbench(folder, judged=False)
    env = dict(os.environ, ORM_RESULTS=str(folder.parent), ORM_V1_DIR=str(score.DATA / "longbench"),
               ORM_MODEL="openai.gpt-oss-120b-1:0", ORM_REGION="ap-south-1", ORM_WORKERS="64",
               ORM_REASONING_EFFORT="")
    subprocess.run([sys.executable, str(HERE / "longbench/verify_v1_orm.py")], env=env, check=True)
    print(json.dumps(score.longbench(folder, judged=True), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    m = sub.add_parser("merge", help="merge FSDP shards using the training environment")
    m.add_argument("actor", type=Path)
    m.add_argument("model", type=Path)
    g = sub.add_parser("generate", help="generate LongBench and LongReason with single-GPU replicas")
    g.add_argument("model", type=Path)
    for cmd in (g, sub.add_parser("judge", help="score unresolved LongBench answers via AWS Bedrock"),
                sub.add_parser("report", help="validate and score both long-context suites offline")):
        cmd.add_argument("results", type=Path)
    args = p.parse_args()
    if args.action == "merge":
        merge(args.actor.resolve(), args.model.resolve())
        return
    root = args.results.resolve()
    if args.action != "generate" and not root.is_dir():
        p.error(f"Results directory does not exist: {root}")
    with lock(root):
        if args.action == "generate":
            asyncio.run(generate(args.model.resolve(), root))
        elif args.action == "judge":
            judge(root)
        else:
            report = json.dumps(score.table(root), indent=2)
            (root / "scores.json").write_text(report + "\n")
            print(report)


if __name__ == "__main__":
    main()
