#!/usr/bin/env python3
"""Stage 0: materialize LongReason tiers to local JSONL, so stage 1 needs no network.

Why a stage 0 at all (same reason as the GSM-Infinite package): the eval host is
often credential/network limited, and the 128k tier is ~760 MB of text -- you
want it staged once and rsync'd, not re-downloaded per model.

Three LongReason-specific things happen here:

1. **Revision pinning.** The upstream card says the pre-2025-05-22 revision
   "contained a bug that caused the ground-truth answer to be misaligned with
   the question". We pin --revision to the fixed commit by default and record it
   in the manifest, so a slice can never be silently the broken one.

2. **Option parsing.** Scoring is exact-letter, so the option list is part of the
   data contract, not of the generation code. We parse `A. ... B. ...` out of the
   final inquiry once, here, and assert the gold letter is in range. Option count
   is NOT always 4 (2/3/4/5 all occur), so the random baseline is
   mean(1/n_options), not 0.25.

3. **Category labelling.** LongReason ships no category column, but the rows are
   stored in category blocks (paper S3.2: 280 reading comprehension / 347 logical
   inference / 167 math word problems, and the row order matches those counts
   exactly -- see verify_longreason_contract.py for the marker-rate evidence).
   We attach the label here so summarize can break results down by category.
   The blocks are INFERRED, not published; the verify script re-checks them.

Also: --sample-k takes an evenly strided subset, never a head slice, because a
head slice of LongReason is 100% reading comprehension.

Usage:
  python3 prepare_longreason.py --out $EVAL_ROOT/eval_data/longreason \
    --tiers 32k 64k 128k
  python3 prepare_longreason.py --out ... --tiers original 32k --sample-k 200
"""
import argparse, json, hashlib, re, sys, urllib.request
from pathlib import Path

HF_DATASET = "lz1bytedance/LongReason"
# fixed-answer revision (dataset card update 2025.5.22); pinned so a slice is
# never silently built from the misaligned-gold version
DEFAULT_REVISION = "2b080b67248d3faa575a143c2d300e6fa1bb21c9"

# parquet shards per split, as published under data/
TIER_FILES = {
    "original": ["original-00000-of-00001.parquet"],
    "expanded": ["expanded-00000-of-00001.parquet"],
    "8k": ["8k-00000-of-00001.parquet"],
    "16k": ["16k-00000-of-00001.parquet"],
    "32k": ["32k-00000-of-00001.parquet"],
    "64k": ["64k-00000-of-00001.parquet"],
    "128k": ["128k-00000-of-00002.parquet", "128k-00001-of-00002.parquet"],
}
ALL_TIERS = list(TIER_FILES)
N_ROWS = 794

# Row-index blocks -> task category. Counts are the paper's (280/347/167) and the
# published row order is grouped by category; verify_longreason_contract.py
# re-derives this from lexical markers before you trust a per-category number.
CATEGORY_BLOCKS = [
    ("reading_comprehension", 0, 280),
    ("logic", 280, 627),
    ("math", 627, 794),
]

# Option lines: every one of the 3,174 options in the dataset uses "X. text";
# ")" is accepted defensively. Options must form a contiguous run from A.
_OPT = re.compile(r"(?m)^[ \t]*([A-E])[.)][ \t]*(.*?)[ \t]*$")


def category_of(qidx):
    for name, lo, hi in CATEGORY_BLOCKS:
        if lo <= qidx < hi:
            return name
    return "unknown"


def parse_options(text):
    """-> [[letter, option_text], ...]; empty if the run is not A,B,C,... ."""
    ms = _OPT.findall(text or "")
    letters = [m[0] for m in ms]
    if not letters or letters != [chr(65 + i) for i in range(len(letters))]:
        return []
    return [[l, t] for l, t in ms]


def fetch_parquet(fname, revision, cache):
    """Download one shard to the parquet cache (skip if already there)."""
    dst = cache / fname
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = (f"https://huggingface.co/datasets/{HF_DATASET}/resolve/"
           f"{revision}/data/{fname}")
    tmp = dst.with_suffix(".part")
    print(f"  downloading {fname} ...", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dst)
    return dst


def iter_rows(files):
    """Stream parquet rows in file/row order (row order is the category order)."""
    import pyarrow.parquet as pq
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=8):
            for row in batch.to_pylist():
                yield row


def keep_indices(n_total, k):
    """Evenly strided subset, so the 280/347/167 category mix is preserved.
    A head slice (rows[:k]) would be pure reading comprehension."""
    if not k or k >= n_total:
        return None
    return {round(i * (n_total - 1) / (k - 1)) if k > 1 else 0 for i in range(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", nargs="+", default=["32k", "64k", "128k"],
                    choices=ALL_TIERS)
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help="HF revision; default = the fixed-gold commit")
    ap.add_argument("--sample-k", type=int, default=0,
                    help="keep k evenly strided questions (0 = all 794)")
    ap.add_argument("--parquet-cache", default=None,
                    help="where shards are downloaded (default <out>/_parquet)")
    ap.add_argument("--drop-parquet", action="store_true",
                    help="delete the shard cache after writing JSONL")
    args = ap.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        sys.exit("pip install pyarrow  (used to read the published parquet shards)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.parquet_cache) if args.parquet_cache else out / "_parquet"
    cache.mkdir(parents=True, exist_ok=True)
    keep = keep_indices(N_ROWS, args.sample_k)

    manifest = {"dataset": HF_DATASET, "revision": args.revision,
                "config": {k: v for k, v in vars(args).items()},
                "category_blocks": [[n, lo, hi] for n, lo, hi in CATEGORY_BLOCKS],
                "tiers": {}}

    for tier in args.tiers:
        dst = out / f"{tier}.jsonl"
        if dst.exists():
            n = sum(1 for _ in dst.open())
            print(f"[{tier}] SKIP (exists, {n} rows)", flush=True)
            manifest["tiers"][tier] = {"n": n, "skipped_existing": True}
            continue
        print(f"[{tier}] staging", flush=True)
        files = [fetch_parquet(f, args.revision, cache) for f in TIER_FILES[tier]]

        tmp = dst.with_suffix(".part")
        n_written = n_seen = 0
        bad_opts, bad_gold, cat_count = [], [], {}
        chars, opt_counts = [], []
        with tmp.open("w") as fh:
            for qidx, row in enumerate(iter_rows(files)):
                n_seen += 1
                if keep is not None and qidx not in keep:
                    continue
                is_long = tier != "original"
                inquiry = row["final_inquiry"] if is_long else row["question"]
                opts = parse_options(inquiry)
                if not opts:
                    bad_opts.append(qidx)
                if not opts or ord(row["answer"]) - 65 >= len(opts):
                    bad_gold.append(qidx)
                rec = {
                    "tier": tier,
                    "qidx": qidx,
                    "example_idx": row["example_idx"],
                    "category": category_of(qidx),
                    "answer": row["answer"],
                    "options": opts,
                    "n_options": len(opts),
                    # background is "" for the original tier
                    "background": row["background"] if is_long else "",
                    "final_inquiry": inquiry,
                    # sha1 of the official rendered prompt: lets stage 1 prove its
                    # `official` reconstruction is byte-exact without us storing a
                    # second copy of a 100k-token context
                    "prompt_sha1": hashlib.sha1(row["prompt"].encode()).hexdigest(),
                    "prompt_chars": len(row["prompt"]),
                }
                cat_count[rec["category"]] = cat_count.get(rec["category"], 0) + 1
                chars.append(len(row["prompt"]))
                opt_counts.append(len(opts))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
        tmp.rename(dst)

        if n_seen != N_ROWS:
            print(f"  WARNING {tier}: read {n_seen} rows, expected {N_ROWS}")
        # random baseline = mean(1/n_options); unparsed rows count as 0 chance
        rb = sum(1 / c if c else 0.0 for c in opt_counts) / n_written
        chars.sort()
        manifest["tiers"][tier] = {
            "n": n_written, "n_upstream": n_seen, "categories": cat_count,
            "random_baseline": round(100 * rb, 4),
            "prompt_chars_p50": chars[len(chars) // 2],
            "prompt_chars_max": chars[-1],
            "unparsed_options": bad_opts, "gold_out_of_range": bad_gold,
        }
        print(f"[{tier}] n={n_written} cats={cat_count} "
              f"random={100*rb:.2f}% prompt_chars p50={chars[len(chars)//2]} "
              f"max={chars[-1]}"
              + (f"  BAD_OPTIONS={len(bad_opts)}" if bad_opts else "")
              + (f"  BAD_GOLD={len(bad_gold)}" if bad_gold else ""), flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.drop_parquet:
        for f in cache.glob("*.parquet"):
            f.unlink()
        print("parquet cache dropped")
    print(f"TOTAL tiers={len(manifest['tiers'])} -> {out}")
    print("NOTE: token budgets are per tier -- check the capacity table in "
          "README before choosing --max-model-len.")


if __name__ == "__main__":
    main()
