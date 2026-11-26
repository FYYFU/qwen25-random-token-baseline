"""Compute the two step-41 table metrics from complete sample records."""
import gzip
import json
from pathlib import Path
from statistics import mean
from longreason.rescore_lenient import lenient_letter

TASKS = ("2wikimqa", "hotpotqa", "musique", "narrativeqa", "qasper")
TIERS = ("8k", "16k", "32k", "64k")
DATA = Path(__file__).resolve().parents[1] / "data/eval"

def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)

def check_keys(actual, expected, name):
    if actual != expected:
        raise ValueError(f"{name}: missing {len(expected-actual)} or unexpected {len(actual-expected)} keys")

def longbench(directory, *, judged):
    generated = {}
    expected = {(t, q, s) for t in TASKS for q in range(200) for s in range(5)}
    inputs = {t: list(records(DATA / "longbench" / f"{t}.jsonl")) for t in TASKS}
    for r in records(directory / "records_v1.jsonl"):
        key = (r["task"], r["qidx"], r["sample"])
        if key not in expected or key in generated:
            raise ValueError(f"Unexpected or duplicate generation {key}")
        source = inputs[key[0]][key[1]]
        if r["question"] != source["input"][:500] or r["gold"] != source["answers"]:
            raise ValueError(f"LongBench input mismatch at {key}")
        if r["raw_tail"].startswith("ERROR "):
            raise ValueError(f"Failed generation at {key}")
        generated[key] = r["acc"]
    check_keys(generated.keys(), expected, "LongBench generations")
    if not judged:
        return
    scores = {}
    for r in records(directory / "verify_records.jsonl"):
        key = (r["task"], r["_qidx"], r["_sample"])
        if key not in expected or key in scores or r["judge_unresolved"]:
            raise ValueError(f"Invalid or unresolved judgment {key}")
        if r["sub_em"] != generated[key] or r["final"] != max(generated[key], r["judge_gpt"] or 0):
            raise ValueError(f"Generation and judgment disagree at {key}")
        scores[key] = r["final"]
    check_keys(scores.keys(), expected, "LongBench judgments")
    return {t: 100 * mean(scores[t,q,s] for q in range(200) for s in range(5)) for t in TASKS}

def longreason(directory, tier):
    expected = {(q,s) for q in range(794) for s in range(5)}
    with (DATA / "longreason" / f"{tier}.jsonl").open() as stream:
        inputs = {r["qidx"]: (r["answer"], r["n_options"]) for r in map(json.loads, stream)}
    scores = {}
    for r in records(directory / "records_longreason.jsonl.gz"):
        key = (r["qidx"], r["sample"])
        if key not in expected or key in scores or r["tier"] != tier or r["error"]:
            raise ValueError(f"Invalid LongReason generation {tier} {key}")
        if inputs[key[0]] != (r["gold"], r["n_options"]):
            raise ValueError(f"LongReason input mismatch at {tier} {key}")
        if r["over_budget"]:
            scores[key] = False
        else:
            options = [(chr(65+i), "") for i in range(r["n_options"])]
            pred, _ = lenient_letter(r["text"], options)
            scores[key] = pred == r["gold"]
    check_keys(scores.keys(), expected, tier)
    return 100 * mean(scores.values())

def table(root):
    qa = longbench(root / "longbench/model", judged=True)
    lr = {t: longreason(root / "longreason" / t, t) for t in TIERS}
    return {name: {k: round(v, 2) for k, v in {**values, "avg": mean(values.values())}.items()}
            for name, values in (("LongBench QA", qa), ("LongReason", lr))}
