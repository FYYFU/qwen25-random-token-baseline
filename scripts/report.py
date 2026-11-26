"""Print the step timings, optimizer evidence, and sampled GPU peaks for a run."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path, help="the directory printed by run.sh")
root = parser.parse_args().run
status = root / "exit_code"
print("Launcher exit code:", status.read_text().strip() if status.exists() else "pending")
metrics_files = list(root.glob("*/metrics.jsonl"))
if len(metrics_files) != 1:
    raise SystemExit("Expected one trainer metrics file under the run directory.")
steps = [json.loads(line) for line in metrics_files[0].read_text().splitlines() if line.strip()]
if not steps:
    raise SystemExit("No completed training steps in metrics.jsonl.")
print("step  seconds  kept_rollouts  skipped  grad_norm")
for row in steps:
    d = row["data"]
    print(f"{row['step']:4}  {d['timing_s/step']:7.2f}  {d['reward_rejection/kept_rows']:13.0f}"
          f"  {d['reward_rejection/skip_update']:7.0f}  {d.get('actor/grad_norm', 0):9.4g}")
print(f"Mean step time: {mean(s['data']['timing_s/step'] for s in steps):.2f} s")
for rank in range(4):
    path = root / f"optimizer_rank{rank}.jsonl"
    probes = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    print(f"Rank {rank}: {len(probes)} optimizer call(s), "
          f"{sum(r['before'] != r['after'] for r in probes)} with a measured parameter change.")
peaks = {}
with (root / "gpu.csv").open() as f:
    for row in csv.DictReader(f):
        gpu = int(row["gpu"])
        trial, device = peaks.get(gpu, (0, 0))
        peaks[gpu] = max(trial, float(row["trial_MiB"])), max(device, float(row["device_MiB"]))
if len(peaks) != 4:
    raise SystemExit("Incomplete GPU sampling. See monitor.log.")
print("GPU  job peak (GiB)  device peak (GiB)")
for gpu, (trial, device) in sorted(peaks.items()):
    print(f"{gpu:3}  {trial / 1024:14.2f}  {device / 1024:17.2f}")
