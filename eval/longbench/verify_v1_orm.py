#!/usr/bin/env python3
"""QwenLong-L1 style ORM verification of existing v1_32k records (no regeneration).

Method (Tongyi-Zhiwen/Qwen-Doc QwenLong-L1/eval/*_verify.py):
  final metric = mean over samples of max(sub_em, judge)
GENERAL_ORM_PROMPT and the user template are copied verbatim; judge =
Bedrock GPT-OSS-120B (temperature 0). Optimization: judge is only called
where sub_em==0 and pred is non-empty (max() makes other calls irrelevant).

Questions are recovered by aligning records (written in task order, map order
preserved) with the v1 dataset files.

Incremental: appends to verify_records.jsonl per model; reruns skip done ids.
"""
import json, os, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config

if not os.environ.get("ORM_RESULTS") or not os.environ.get("ORM_V1_DIR"):
    sys.exit("set ORM_RESULTS=<stage-1 output dir> and ORM_V1_DIR=<longbench_v1_subset dir>")
RESULTS = Path(os.environ["ORM_RESULTS"])
V1_DIR = Path(os.environ["ORM_V1_DIR"])
# Default judge = gpt-oss-120b (user decision 2026-08-10; dual-judge check vs
# claude-sonnet-4.5 agreed within ~0.01 RM with identical rankings).
JUDGE_MODEL = os.environ.get("ORM_MODEL", "openai.gpt-oss-120b-1:0")
REGION = os.environ.get("ORM_REGION", "ap-south-1")
# gpt-oss judges accept {"reasoning_effort": low|medium|high} (default medium).
# Leave unset for Claude judges.
REASONING_EFFORT = os.environ.get("ORM_REASONING_EFFORT", "")
WORKERS = int(os.environ.get("ORM_WORKERS", "8"))

GENERAL_ORM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.
Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

_client = None

def client():
    global _client
    if _client is None:
        # A fresh Session (NOT boto3.client, which reuses the module-level
        # DEFAULT_SESSION) so rebuilt clients re-resolve credentials after an
        # ExpiredToken. With the default session, "rebuilding" kept the stale
        # credentials forever (observed 2026-07-31).
        _client = boto3.session.Session().client(
            "bedrock-runtime", region_name=REGION,
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
    return _client


def judge_call(problem, pred, gold):
    global _client
    prompt = ORM_USER_TEMPLATE.format(problem=problem, answer_1=pred, answer_2=gold)
    backoff = 2
    for attempt in range(12):
        try:
            kwargs = dict(
                modelId=JUDGE_MODEL,
                system=[{"text": GENERAL_ORM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 512, "temperature": 0.0},
            )
            if REASONING_EFFORT:
                kwargs["additionalModelRequestFields"] = {"reasoning_effort": REASONING_EFFORT}
            resp = client().converse(**kwargs)
            # join all text blocks: some models (e.g. gpt-oss) prepend a
            # reasoningContent block with no "text" key
            text = " ".join(
                b["text"] for b in resp["output"]["message"]["content"] if "text" in b)
            return 1.0 if "YES" in text.upper() else 0.0
        except Exception as e:
            msg = str(e)
            if "ExpiredToken" in msg or "InvalidClientTokenId" in msg:
                # wait for external credential refresh, then rebuild client
                print(f"[creds] expired; waiting 120s (attempt {attempt})", flush=True)
                time.sleep(120)
                _client = None
            else:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
    return None  # unresolved -> counted as 0, flagged in output


def load_questions():
    q = {}
    for tf in sorted(V1_DIR.glob("*.jsonl")):
        rows = [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
        q[tf.stem] = [r["input"] for r in rows]
    return q


def main():
    questions = load_questions()
    models = sorted(d.name for d in RESULTS.iterdir() if (d / "records_v1.jsonl").exists())
    print("models:", models, flush=True)

    for m in models:
        rec_f = RESULTS / m / "records_v1.jsonl"
        out_f = RESULTS / m / "verify_records.jsonl"
        recs = [json.loads(l) for l in rec_f.read_text().splitlines() if l.strip()]
        # question alignment: explicit qidx/sample (Avg@N records) or fall
        # back to per-task write order (legacy n=1 records)
        counters = defaultdict(int)
        for r in recs:
            r["_qidx"] = r.get("qidx", counters[r["task"]])
            r["_sample"] = r.get("sample", 0)
            counters[r["task"]] += 1
        done = set()
        if out_f.exists():
            # A stopped job may leave an unfinished last line. Retry unresolved
            # judgments and keep successful keys exactly once.
            clean = out_f.with_suffix(".tmp")
            with out_f.open() as old, clean.open("w") as kept:
                for line in old:
                    if not line.endswith("\n") or not line.strip():
                        continue
                    item = json.loads(line)
                    key = (item["task"], item["_qidx"], item["_sample"])
                    if key not in done and not item["judge_unresolved"]:
                        kept.write(line)
                        done.add(key)
            clean.replace(out_f)
        todo = [r for r in recs if (r["task"], r["_qidx"], r["_sample"]) not in done]
        print(f"== {m}: {len(recs)} records, {len(done)} done, {len(todo)} todo", flush=True)

        def one(r):
            sub_em = float(r["acc"])
            pred = (r.get("pred") or "").strip()
            if sub_em >= 1.0:
                jg, final = None, 1.0            # judge irrelevant under max()
            elif not pred:
                jg, final = None, 0.0            # nothing to judge
            else:
                problem = questions[r["task"]][r["_qidx"]]
                jg = judge_call(problem, pred, r["gold"][0])
                final = max(sub_em, jg or 0.0)
            return {"task": r["task"], "_qidx": r["_qidx"], "_sample": r["_sample"],
                    "sub_em": sub_em, "judge_gpt": jg, "final": final,
                    "judge_unresolved": int(jg is None and not sub_em and bool(pred))}

        n_done = 0
        with out_f.open("a") as fout, ThreadPoolExecutor(WORKERS) as ex:
            for res in ex.map(one, todo):
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 100 == 0:
                    print(f"[{m}] {n_done}/{len(todo)}", flush=True)

        # summarize
        allrows = [json.loads(l) for l in out_f.read_text().splitlines() if l.strip()]
        agg = defaultdict(list)
        for x in allrows:
            agg[x["task"]].append(x["final"])
        summ = {t: round(sum(v) / len(v), 4) for t, v in sorted(agg.items())}
        summ["avg_rm"] = round(sum(sum(v) / len(v) for v in agg.values()) / len(agg), 4)
        summ["avg_sub_em"] = round(sum(x["sub_em"] for x in allrows) / len(allrows), 4)
        summ["judged"] = sum(1 for x in allrows if x["judge_gpt"] is not None)
        summ["unresolved"] = sum(x["judge_unresolved"] for x in allrows)
        summ["n"] = len(allrows)
        (RESULTS / m / "summary_verify.json").write_text(json.dumps(summ, indent=2))
        print(f"== {m} SUMMARY {json.dumps(summ)}", flush=True)

    print("ALL_VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
