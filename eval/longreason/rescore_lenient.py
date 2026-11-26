#!/usr/bin/env python3
"""Post-hoc diagnostic: separate "cannot answer" from "cannot format".

The frozen protocol scores the training contract strictly: `</think>` must close
and the answer must sit in `\\boxed{}` (SPEC section 4). On LongReason the 3B arms
mostly emit `<think>...</think>\\n<answer>D</answer>` instead, so a strict acc near
0 is measuring contract compliance, not reasoning. This script reads the records
a completed unit already wrote and adds a lenient column, so the main table can
report acc / fmt_ok / acc_lenient together.

It does NOT touch `summary_longreason.json`: the frozen number and the queue's
completeness gate stay exactly as they were. Output goes to
`summary_lenient.json` next to the records.

Lenient extraction reads `raw_tail` (the last 300 characters of the completion,
which is where the answer lands) and accepts, in order:
  1. `<answer> X </answer>`      the format the 3B arms drifted to
  2. `\\boxed{X}` anywhere        boxed without a closed `</think>`
  3. the existing letter parser   "Answer: B", a bare letter, verbatim option text

`answer_tag_rate` and `tail_truncated_rate` are reported alongside so a lenient
number is never read without the reason the strict one failed.

Usage:
  python3 rescore_lenient.py <results_dir> [--write]     # all units
  python3 rescore_lenient.py <results_dir>/<model>/<tier> [--write]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_longreason import letter_from_text  # noqa: E402

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_OPEN_ANSWER_TAG = re.compile(r"<answer>(.*)", re.S | re.I)
_BOXED_ANY = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def lenient_letter(tail: str, options) -> tuple[str, str]:
    """-> (letter, which rule fired)"""
    if not tail:
        return "", "empty"
    m = _ANSWER_TAG.findall(tail)
    if m:
        letter = letter_from_text(m[-1].strip(), options)
        if letter:
            return letter, "answer_tag"
    m = _OPEN_ANSWER_TAG.search(tail)  # tail cut mid-tag
    if m:
        letter = letter_from_text(m.group(1).strip(), options)
        if letter:
            return letter, "answer_tag_open"
    m = _BOXED_ANY.findall(tail)
    if m:
        letter = letter_from_text(m[-1].strip(), options)
        if letter:
            return letter, "boxed_anywhere"
    letter = letter_from_text(tail.strip().splitlines()[-1] if tail.strip() else "", options)
    if letter:
        return letter, "trailing_text"
    return "", "none"


def _read_records(unit: Path) -> list[dict] | None:
    """Records are gzipped when full completions are stored (the default since
    2026-09-06); plain .jsonl stays readable for the earlier runs."""
    gz, plain = (unit / "records_longreason.jsonl.gz", unit / "records_longreason.jsonl")
    if gz.is_file():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            text = fh.read()
    elif plain.is_file():
        text = plain.read_text()
    else:
        return None
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def rescore_unit(unit: Path) -> dict | None:
    summary_path = unit / "summary_longreason.json"
    rows = _read_records(unit)
    if rows is None or not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if not rows:
        return None

    # options are not stored per record; recover them from the staged slice when
    # available, otherwise fall back to plain letters
    n_lenient = 0
    rules: dict[str, int] = {}
    answer_tag = 0
    for row in rows:
        if row["fmt_ok"]:
            n_lenient += row["acc"]
            rules["strict"] = rules.get("strict", 0) + 1
            continue
        if row.get("over_budget"):
            rules["over_budget"] = rules.get("over_budget", 0) + 1
            continue
        options = [(chr(65 + i), "") for i in range(row["n_options"])]
        # full completion when the run stored it, else the 300-character tail
        # (which makes the lenient number a lower bound -- see unrecovered_rate)
        body = row.get("text") or row.get("raw_tail", "")
        letter, rule = lenient_letter(body, options)
        rules[rule] = rules.get(rule, 0) + 1
        if "<answer>" in body.lower():
            answer_tag += 1
        if letter and letter == row["gold"]:
            n_lenient += 1

    n = len(rows)
    out = {
        "model": unit.parent.name,
        "tier": summary["tier"],
        "contract": summary["config"]["contract"],
        "n": n,
        "acc_strict": summary["acc"],
        "fmt_ok_rate": summary["fmt_ok_rate"],
        "acc_lenient": round(n_lenient / n, 4),
        "answer_tag_rate": round(answer_tag / n, 4),
        "unrecovered_rate": round(rules.get("none", 0) / n, 4),
        "rules": rules,
        "note": (
            "acc_lenient is a diagnostic read off the stored 300-character tail, "
            "not the frozen protocol number"
        ),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="results dir, or a single <model>/<tier> dir")
    parser.add_argument("--write", action="store_true",
                        help="also write summary_lenient.json into each unit dir")
    args = parser.parse_args()

    root = Path(args.path)
    if (root / "records_longreason.jsonl").is_file() or (root / "records_longreason.jsonl.gz").is_file():
        units = [root]
    else:
        units = sorted({p.parent for p in list(root.glob("*/*/records_longreason.jsonl"))
                        + list(root.glob("*/*/records_longreason.jsonl.gz"))})
    results = []
    for unit in units:
        out = rescore_unit(unit)
        if out is None:
            continue
        results.append(out)
        if args.write:
            (unit / "summary_lenient.json").write_text(json.dumps(out, indent=2))

    if not results:
        raise SystemExit(f"no completed units under {root}")
    header = ("model", "tier", "acc", "fmt_ok", "acc_len", "ans_tag", "unrec")
    print("%-44s %-5s %-7s %-7s %-8s %-8s %s" % header)
    for row in sorted(results, key=lambda r: (r["tier"], r["model"])):
        print("%-44s %-5s %-7.4f %-7.3f %-8.4f %-8.3f %.3f" % (
            row["model"], row["tier"], row["acc_strict"], row["fmt_ok_rate"],
            row["acc_lenient"], row["answer_tag_rate"], row["unrecovered_rate"]))
    print(f"\n{len(results)} units. acc = frozen protocol; acc_len = diagnostic "
          f"(lenient extraction off the stored tail).")


if __name__ == "__main__":
    main()
