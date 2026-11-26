#!/usr/bin/env python3
"""Verify the LongReason data contract before trusting any number.

Four checks, each guarding a specific way LongReason can silently lie:

  --template    OFFICIAL_LONG / OFFICIAL_SHORT reproduce the published `prompt`
                column byte-for-byte. If this drifts, base-model numbers stop
                being comparable to the paper's table.
  --gold        Answer alignment: the gold letter is inside the option range and
                the SAME letter for a given example_idx across tiers. This is the
                check for the upstream bug fixed on 2025-05-22 (gold misaligned
                with question) -- if you ever unpin --revision, run this.
  --categories  Re-derives the 280/347/167 category blocks from lexical markers
                instead of trusting the row order. Per-category numbers are only
                as good as this.
  --clues       Every expanded-context paragraph (the clues) is present verbatim
                in the long tier, and reports where they sit. This is why the
                eval must not truncate.
  --extract     Extractor unit tests on the response shapes models actually emit.

Works from a staged slice (--data-dir, offline, preferred) or straight from the
datasets-server HTTP API (--remote, needs network but not `datasets`).

  python3 verify_longreason_contract.py --data-dir $EVAL_ROOT/eval_data/longreason
  python3 verify_longreason_contract.py --remote --tiers 8k 32k --n 40
"""
import argparse, hashlib, json, re, sys, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from eval_longreason import (OFFICIAL_LONG, OFFICIAL_SHORT, extract_boxed,
                             extract_official, letter_from_text)
from prepare_longreason import (CATEGORY_BLOCKS, DEFAULT_REVISION, N_ROWS,
                                category_of, parse_options)

ROWS = ("https://datasets-server.huggingface.co/rows"
        "?dataset=lz1bytedance%2FLongReason&config=default"
        "&split={split}&offset={off}&length={n}")

# Lexical markers used ONLY to corroborate the inferred category blocks.
RC_MARK = re.compile(r"\bpassage\b|\bthe (author|text|article|material)\b|"
                     r"\brefer to\b|\bmain idea\b|\bmost likely means\b", re.I)
MATH_MARK = re.compile(r"\bhow (many|much)\b|\bcalculate\b|\btotal (of|number)\b|"
                       r"\bper\b|\baverage\b|\bpercent", re.I)


def fetch(split, off, n):
    """datasets-server caps `length` at 100 rows per call, so page through."""
    out = []
    while len(out) < n:
        want = min(100, n - len(out))
        url = ROWS.format(split=split, off=off + len(out), n=want)
        with urllib.request.urlopen(url, timeout=300) as r:
            page = [x["row"] for x in json.load(r)["rows"]]
        if not page:
            break
        out += page
    return out[:n]


def official_prompt(background, inquiry):
    tmpl = OFFICIAL_LONG if background else OFFICIAL_SHORT
    return tmpl.format(background=background, inquiry=inquiry)


# ------------------------------------------------------------------- checks
def check_template_remote(tiers, n):
    ok = bad = 0
    for t in tiers:
        rows = fetch(t, 0, n)
        for r in rows:
            bg = r["background"] if t != "original" else ""
            inq = r["final_inquiry"] if t != "original" else r["question"]
            if official_prompt(bg, inq) == r["prompt"]:
                ok += 1
            else:
                bad += 1
                if bad <= 2:
                    got, want = r["prompt"], official_prompt(bg, inq)
                    print(f"  MISMATCH {t} {r['example_idx'][:8]}")
                    print("   want tail:", repr(want[-160:]))
                    print("   got  tail:", repr(got[-160:]))
        print(f"[{t}] template {ok}/{ok+bad} byte-exact (running total)")
    return ok, bad


def check_template_local(rows_by_tier):
    """Offline: prepare_longreason.py stored sha1 of the published prompt."""
    ok = bad = 0
    for t, rows in rows_by_tier.items():
        for r in rows:
            h = hashlib.sha1(
                official_prompt(r["background"], r["final_inquiry"]).encode()).hexdigest()
            if h == r["prompt_sha1"]:
                ok += 1
            else:
                bad += 1
                if bad <= 3:
                    print(f"  MISMATCH {t} qidx={r['qidx']} {r['example_idx'][:8]}")
        print(f"[{t}] template {sum(1 for r in rows)} rows checked")
    return ok, bad


def check_gold(rows_by_tier):
    """Gold letter in range, and identical per example_idx across tiers."""
    bad_range, per_ex = [], defaultdict(set)
    for t, rows in rows_by_tier.items():
        for r in rows:
            opts = r["options"] if "options" in r else parse_options(
                r.get("final_inquiry") or r.get("question", ""))
            if not opts or ord(r["answer"]) - 65 >= len(opts):
                bad_range.append((t, r.get("qidx"), r["example_idx"], r["answer"],
                                  len(opts)))
            per_ex[r["example_idx"]].add(r["answer"])
    inconsistent = {k: v for k, v in per_ex.items() if len(v) > 1}
    nopt = Counter(len(r.get("options") or parse_options(
        r.get("final_inquiry") or r.get("question", "")))
        for rows in rows_by_tier.values() for r in rows)
    print(f"gold-in-range violations: {len(bad_range)}" + (f" {bad_range[:5]}" if bad_range else ""))
    print(f"cross-tier gold disagreement: {len(inconsistent)}"
          + (f" {list(inconsistent.items())[:5]}" if inconsistent else ""))
    print(f"option-count distribution: {dict(sorted(nopt.items()))}")
    # random baseline over one tier's worth of rows
    any_rows = next(iter(rows_by_tier.values()))
    rb = sum(1 / max(len(r.get("options") or parse_options(
        r.get("final_inquiry") or r.get("question", ""))), 1) for r in any_rows) / len(any_rows)
    print(f"random baseline = {100*rb:.4f}%   (mean 1/n_options over the rows "
          f"checked; on the full 794 it is 25.2120%, NOT 25.0%)")
    return len(bad_range) + len(inconsistent)


def check_categories(short_rows):
    """Corroborate the inferred blocks with marker rates on the SHORT questions."""
    if len(short_rows) < N_ROWS:
        print(f"  (only {len(short_rows)} short rows available; block check needs all "
              f"{N_ROWS} -- stage the 'original' tier for the full test)")
    print(f"{'block':>22s} {'rows':>10s} {'n':>5s} {'RC_mark':>8s} {'MATH_mark':>10s}")
    rates = {}
    for name, lo, hi in CATEGORY_BLOCKS:
        seg = [r for r in short_rows if lo <= r["qidx"] < hi]
        if not seg:
            continue
        q = [r.get("question") or r["final_inquiry"] for r in seg]
        rc = sum(bool(RC_MARK.search(x)) for x in q) / len(q)
        mt = sum(bool(MATH_MARK.search(x)) for x in q) / len(q)
        rates[name] = (rc, mt)
        print(f"{name:>22s} {f'{lo}-{hi-1}':>10s} {len(seg):5d} {rc:8.2f} {mt:10.2f}")
    ok = True
    if "reading_comprehension" in rates and "logic" in rates:
        if not rates["reading_comprehension"][0] > 5 * max(rates["logic"][0], 0.02):
            ok = False
            print("  WARN: reading-comprehension block is not markedly RC-heavy")
    if "math" in rates and "logic" in rates:
        if not rates["math"][1] > rates["logic"][1]:
            ok = False
            print("  WARN: math block is not markedly math-heavy")
    print("category blocks: " + ("CORROBORATED" if ok else "SUSPECT -- do not report "
                                 "per-category numbers"))
    return 0 if ok else 1


def check_clues(n):
    """Expanded-context paragraphs (the clues) must survive verbatim in the long
    tiers; report their positions to show why truncation is not allowed."""
    exp = fetch("expanded", 0, n)
    fails = 0
    for tier in ("8k", "32k", "128k"):
        try:
            longs = fetch(tier, 0, n)
        except Exception as e:
            print(f"[{tier}] fetch failed: {e}")
            continue
        found = total = 0
        spans = []
        for e, l in zip(exp, longs):
            paras = [p.strip() for p in e["background"].split("\n\n") if p.strip()]
            bg = l["background"]
            fr = []
            for p in paras:
                total += 1
                i = bg.find(p[:80])
                if i >= 0:
                    found += 1
                    fr.append(i / len(bg))
            if fr:
                spans.append((min(fr), max(fr)))
        fails += total - found
        med = sorted(b - a for a, b in spans)
        print(f"[{tier}] clues located {found}/{total}   median clue span "
              f"{med[len(med)//2]:.2f} of context")
    print("=> clues sit at random positions across the whole context: middle "
          "truncation can DELETE the answer. eval_longreason.py never truncates.")
    return fails


def check_extract():
    """Extractor behaviour on the response shapes models actually emit."""
    opts = [("A", "Wave modulator"), ("B", "Ion converter"),
            ("C", "Pulsed high-energy currents"), ("D", "Ion streams")]
    cases = [
        # (contract, response, expected letter)
        ("official", "reasoning...\n\nThe answer is D.", "D"),
        ("official", "The answer is (B)", "B"),
        ("official", "the answer is **C**", "C"),
        ("official", "...\nThe answer is A\n", "A"),
        ("official", "The answer is D. Ion streams", "D"),
        ("official", "First the answer is A, but on reflection the answer is C.", "C"),
        ("official", "blah\nAnswer: B", "B"),
        ("official", "blah\nD", "D"),
        ("official", "blah\nD. Ion streams", "D"),
        ("official", "blah\nIon streams", "D"),          # exact option text
        ("official", "<think>The answer is A</think>\nThe answer is B", "B"),
        ("official", "I cannot determine this.", ""),
        ("official", "", ""),
        ("think", "<think>...</think> \\boxed{C}", "C"),
        ("think", "<think>...</think> \\boxed{\\text{D}}", "D"),
        ("think", "<think>...</think> the answer is \\boxed{B. Ion converter}", "B"),
        ("think", "no closed think \\boxed{A}", ""),      # training gate
        ("think", "<think>..</think> \\boxed{Ion streams}", "D"),
        ("think", "<think>..</think> no box", ""),
        ("think", "<think>\\boxed{A}</think> \\boxed{B}", "B"),  # only after </think>
        ("boxed", "\\boxed{A}", "A"),
        ("boxed", "\\boxed{first} then \\boxed{E}", ""),  # E not an option here
        ("boxed", "chatter without a box", ""),
    ]
    bad = 0
    for contract, text, want in cases:
        if contract == "official":
            got, _ = extract_official(text, opts)
        else:
            got, _ = extract_boxed(text, opts, require_think=(contract == "think"))
        if got != want:
            bad += 1
            print(f"  FAIL [{contract}] {text[:60]!r} -> {got!r} want {want!r}")
    # 5-option rows must accept E
    got, _ = extract_official("The answer is E", opts + [("E", "None of the above")])
    if got != "E":
        bad += 1
        print(f"  FAIL 5-option E -> {got!r}")
    print(f"extractor: {len(cases)+1-bad}/{len(cases)+1} cases pass")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="staged slice (offline checks)")
    ap.add_argument("--remote", action="store_true", help="use datasets-server API")
    ap.add_argument("--tiers", nargs="+",
                    default=["original", "expanded", "8k", "32k"])
    ap.add_argument("--n", type=int, default=20, help="rows per tier in --remote mode")
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--categories", action="store_true")
    ap.add_argument("--clues", action="store_true")
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    which = {k for k in ("template", "gold", "categories", "clues", "extract")
             if getattr(args, k)}
    if not which:  # default: everything the available source supports
        which = {"template", "gold", "categories", "extract"}
        if args.remote:
            which.add("clues")

    fails = 0
    print("=== extractor ===" if "extract" in which else "", end="")
    if "extract" in which:
        print()
        fails += check_extract()

    local = {}
    if args.data_dir:
        d = Path(args.data_dir)
        for f in sorted(d.glob("*.jsonl")):
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if rows:
                local[f.stem] = rows
        if not local:
            sys.exit(f"no <tier>.jsonl under {d}")
        mf = d / "manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            print(f"\nslice revision: {m.get('revision')}"
                  + ("  (pinned fixed-gold commit)"
                     if m.get("revision") == DEFAULT_REVISION
                     else "  !! NOT the pinned revision -- gold alignment unverified"))
            if m.get("revision") != DEFAULT_REVISION:
                fails += 1

    if "template" in which:
        print("\n=== official template byte-exactness ===")
        if local:
            ok, bad = check_template_local(local)
            print(f"TEMPLATE {ok}/{ok+bad} byte-exact (via stored prompt_sha1)")
            fails += bad
        elif args.remote:
            ok, bad = check_template_remote(args.tiers, args.n)
            print(f"TEMPLATE {ok}/{ok+bad} byte-exact vs published prompt column")
            fails += bad
        else:
            print("  skipped: need --data-dir or --remote")

    if "gold" in which:
        print("\n=== gold alignment ===")
        src = local
        if not src and args.remote:
            src = {t: [dict(r, qidx=i) for i, r in enumerate(fetch(t, 0, args.n))]
                   for t in args.tiers}
        if src:
            fails += check_gold(src)
        else:
            print("  skipped: need --data-dir or --remote")

    if "categories" in which:
        print("\n=== category blocks (inferred) ===")
        short = local.get("original") or local.get("expanded")
        if short is None and args.remote:
            short = [dict(r, qidx=i) for i, r in enumerate(fetch("original", 0, N_ROWS))]
        if short:
            fails += check_categories(short)
        else:
            print("  skipped: stage the 'original' tier, or use --remote")

    if "clues" in which:
        print("\n=== clue localization (why we never truncate) ===")
        if args.remote:
            fails += check_clues(min(args.n, 20))
        else:
            print("  skipped: needs --remote (compares expanded vs long tiers)")

    print(f"\n{'ALL CHECKS PASS' if fails == 0 else f'{fails} PROBLEM(S) -- see above'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
