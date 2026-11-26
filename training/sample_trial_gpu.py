"""Sample physical-device and trial process memory through NVML."""
import csv
import os
from pathlib import Path
import sys
import time

import pynvml as nv

trial = Path(sys.argv[1]).resolve()
marker = os.fsencode(f"CFOPSDL_TELEMETRY_DIR={trial}")
nv.nvmlInit()
handles = {int(i): nv.nvmlDeviceGetHandleByIndex(int(i))
           for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")}
known = {}

def belongs(pid):
    if pid not in known:
        try:
            proc = Path("/proc") / str(pid)
            if proc.stat().st_uid != os.getuid():
                known[pid] = False
            else:
                known[pid] = marker in (proc / "environ").read_bytes().split(b"\0")
        except OSError:
            return False
    return known[pid]

with (trial / "gpu.csv").open("w", buffering=1) as stream:
    rows = csv.writer(stream)
    rows.writerow(["unix_s", "gpu", "device_MiB", "trial_MiB", "trial_pids"])
    while True:
        now = time.time()
        for i,h in handles.items():
            procs = [p for p in nv.nvmlDeviceGetComputeRunningProcesses(h) if belongs(p.pid)]
            used = sum(p.usedGpuMemory for p in procs if 0 <= p.usedGpuMemory < 2**50)
            rows.writerow([round(now,3), i, round(nv.nvmlDeviceGetMemoryInfo(h).used/2**20,1),
                           round(used/2**20,1), " ".join(str(p.pid) for p in procs)])
        time.sleep(0.25)
