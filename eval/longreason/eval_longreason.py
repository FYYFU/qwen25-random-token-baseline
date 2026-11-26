#!/usr/bin/env python3
"""Stage 1: LongReason generation + rule score, per length tier.

Same two-way decoupling as the other packages in eval/ (see SPEC S2):
  - PROMPT follows the evaluated model's TRAINING contract (--contract), not
    LongReason's own template. `official` reproduces the upstream template
    byte-exactly and is the base-model protocol.
  - SCORING is contract-independent, exact, and needs NO judge: the gold is a
    single option letter. This is the big operational win over LongBench-v1
    (no Bedrock ORM stage, no credentials, results are final at stage 1).

LongReason specifics that shape this file:

* **Per-tier scores.** Each tier (8k/16k/32k/64k/128k) is the SAME 794 questions
  with more irrelevant Pile filler, so tier-to-tier deltas are a clean
  length-degradation curve on fixed content. One --tier per invocation, one
  output dir per (model, tier): tiers need different server windows, so they
  cannot share a vLLM process.

* **Truncation is fatal here, unlike LongBench-v1.** Clues sit at random
  positions (median span 0.20-0.82 of the context) and the whole point is
  aggregating them, so middle-truncating a 128k context can delete the answer.
  We therefore do NOT truncate: a prompt over budget is recorded as
  `over_budget` and scored 0, and the run prints the rate. If over_budget > 0
  you must raise --max-model-len (or admit the model can't hold that tier)
  rather than quietly cut context.

* **Extraction is letter-based.** Upstream takes "the first character following
  the phrase 'the answer is'". We keep that for `official`, and for the trained
  contracts read the letter out of the last \\boxed{} (after `</think>` when the
  contract gates on it). Everything runs through the same
  `letter_from_text` normalizer so "B", "(B)", "B.", "Answer: B", and
  "B. Ion streams" all resolve to B -- and an option-text-only answer is matched
  back to its letter, since a model that names the right option should not be
  punished by the extractor.

Records: one row per sample, carrying (tier, qidx, category, sample) so
summarize_longreason.py can build the tier curve and the per-category table.
"""
import argparse, gzip, hashlib, json, re, statistics, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ------------------------------------------------------------------ contracts
# byte-exact from the training parquets (same strings as the other packages)
BOXED_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and "
    "the Assistant solves it. The Assistant provides the final answer enclosed "
    "within \\boxed{} tags, i.e., \\boxed{answer here}. The \\boxed{} should "
    "contain only the answer itself, with no explanation or extra words."
)
THINK_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and "
    "the Assistant solves it. The Assistant first thinks about the reasoning "
    "process in the mind and then provides the User with the answer. The "
    "reasoning process is enclosed within <think> </think> and answer is "
    "enclosed within \\boxed{} tags, respectively, i.e., <think> reasoning "
    "process here </think> \\boxed{answer here}."
)
# LongReason's own zero-shot CoT prompt. Verified byte-exact against the
# published `prompt` column on all 794 rows of original/expanded/8k/32k/64k/128k
# (verify_longreason_contract.py). Note the details: the trailing period (the
# paper's Table 8/9 uses "!"), the curly quotes around "The answer is", the
# "analyze it only once" sentence, and "### Answer " with a trailing space in the
# short-question variant only.
OFFICIAL_LONG = (
    "### Background Information\n{background}\n\n"
    "### Question about the Background Information\n{inquiry}\n\n"
    "Please answer the above question based on the provided background information!\n\n"
    "### Answer\n"
    "Please analyze the question step by step, and analyze it only once. Do not "
    "analyze the question repeatedly. Provide the final answer on the last line "
    "using \u201cThe answer is\u201d + option (represented by A, B, C, D, E).\n"
)
OFFICIAL_SHORT = (
    "### Question\n{inquiry}\n\n"
    "Please answer the above question!\n\n"
    "### Answer \n"
    "Please analyze the question step by step, and analyze it only once. Do not "
    "analyze the question repeatedly. Provide the final answer on the last line "
    "using \u201cThe answer is\u201d + option (represented by A, B, C, D, E).\n"
)
# Qwen's default system prompt: `official` renders as a user-only chat (matching
# the LongBench-v1 package's base_official protocol).
OFFICIAL_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Trained-contract user turn. The task statement stays; the output-format
# sentence is dropped because the contract's system prompt owns the format.
# The MCQ answer-set instruction stays -- without it the model cannot know it
# must pick a letter, and that is task information, not format compliance.
TRAINED_LONG = (
    "The following are given passages.\n{background}\n\nQuestion: {inquiry}\n\n"
    "Answer with the letter of the correct option."
)
TRAINED_SHORT = (
    "{inquiry}\n\nAnswer with the letter of the correct option."
)

# v2-aligned MCQ envelope (`think_mcq` / `boxed_mcq`).
#
# The sentence above is a paraphrase of the training one with its format half cut
# off. The training mix contains a multiple-choice shard whose user turn ends with
# the sentence below, byte for byte, and whose reward only credits a
# PARENTHESISED letter inside \boxed{} -- so on an MCQ benchmark the paraphrase is
# the misaligned option, not the faithful one. This is the same replacement the
# frozen LongBench-v2 bundle makes (protocol_lbv2.TRAINED_TASK_SENTENCE).
#
#   shard: s3://yufu-2026/0716-rlsd/data/loongrl_mix_16kkc_orz_rlsd_v2numbered/
#          mathqa_choice/
TRAINED_MCQ_SENTENCE = (
    "Answer with the letter of the correct option wrapped in \\boxed{}, "
    "e.g. \\boxed{(a)}."
)
TRAINED_MCQ_SENTENCE_SHA1 = "4dcdcb87210aec564e95a4e6cc8acd8c81c0e577"
# Stems are formatted first and the sentence is appended, never .format()ed: the
# sentence contains literal braces (\boxed{}, \boxed{(a)}) and str.format would
# read them as fields.
TRAINED_LONG_STEM = (
    "The following are given passages.\n{background}\n\nQuestion: {inquiry}\n\n"
)
TRAINED_SHORT_STEM = "{inquiry}\n\n"

# ----------------------------------------------------------------- extraction
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
# upstream: "the first character following the phrase 'the answer is'"; the
# curly-apostrophe / bold / colon variants models actually emit are absorbed here
_ANSWER_IS = re.compile(r"the answer is[\s:*\"'\u201c\u2018\(\[]*([A-Ea-e])")
_BARE_LETTER = re.compile(r"^[\s*\(\[\"'\u201c]*([A-Ea-e])[\s.):\]\"'\u201d,]*$")
_LEAD_LETTER = re.compile(r"^[\s*\(\[\"'\u201c]*([A-Ea-e])[.):\]]")


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split())


def letter_from_text(text, options):
    """Resolve a free-form answer span to an option letter, or "".

    Order matters: an explicit letter wins over option-text matching, so a model
    that says "B. Ion streams" is scored on B even if the text of another option
    happens to be a substring.
    """
    if not text:
        return ""
    t = text.strip()
    valid = {l for l, _ in options} or set("ABCDE")

    m = _BARE_LETTER.match(t) or _LEAD_LETTER.match(t)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    # "Answer: B" / "option C" style prefixes
    m = re.match(r"^\W*(?:the\s+)?(?:answer|option|choice)\W*(?:is\W*)?([A-Ea-e])\b", t, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = _ANSWER_IS.search(t.lower())
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    # option text quoted verbatim (exact normalized match only -- no substring
    # guessing, which would mis-resolve options that share wording)
    n = _norm(t)
    if n:
        exact = [l for l, otext in options if _norm(otext) and _norm(otext) == n]
        if len(exact) == 1:
            return exact[0]
    return ""


def extract_official(text, options):
    """Upstream rule: the character right after the LAST 'the answer is'.
    Last-match (upstream scans for the phrase) is the safe choice for models
    that restate the question. Falls back to the last non-empty line, which is
    where the template told the model to put the letter."""
    if not text:
        return "", ""
    body = text.rsplit("</think>", 1)[1] if "</think>" in text else text
    low = body.lower()
    i = low.rfind("the answer is")
    if i != -1:
        span = body[i:i + 80]
        return letter_from_text(span, options), span.strip()[:120]
    lines = [l for l in body.strip().splitlines() if l.strip()]
    if lines:
        return letter_from_text(lines[-1], options), lines[-1].strip()[:120]
    return "", ""


# Training MCQ reward's identity rule (loongrl_reward.mathqa_choice_compute_score):
#   pred_option = re.search(r"\((\w)\)", pred);  pred_option.group(1).lower() == gold.lower()
# A bare letter scores 0 in training and the arms never emit one (0/171 in the 3B
# step-41 MCQ rollouts), so `*_mcq` does not accept one -- that would credit a form
# RL never rewarded. Same rule as the frozen LongBench-v2 bundle.
_OPTION = re.compile(r"\((\w)\)")


def extract_boxed_option(text, options, require_think):
    """`*_mcq` contracts: parenthesised letter in the last \\boxed{}.

    With require_think, only after a closed `</think>` (the training gate: an
    unclosed think means pred "" and a score of 0). The letter is uppercased even
    when out of range: the model committed to an option, just a wrong one, and
    calling that a parse failure would hide a real error inside fmt_ok.
    """
    if not text:
        return "", ""
    if require_think:
        if "</think>" not in text:
            return "", ""
        text = text.rsplit("</think>", 1)[1]
    m = _BOXED.findall(text)
    if not m:
        return "", ""
    payload = m[-1].strip()
    hit = _OPTION.search(payload)
    return (hit.group(1).upper() if hit else ""), payload[:120]


def extract_boxed(text, options, require_think):
    """Last \\boxed{...} -> letter; with require_think, only AFTER </think>
    (the training reward gate: unclosed think -> "" -> 0, no fallback)."""
    if not text:
        return "", ""
    if require_think:
        if "</think>" not in text:
            return "", ""
        text = text.rsplit("</think>", 1)[1]
    m = _BOXED.findall(text)
    if not m:
        return "", ""
    ans = m[-1].strip()
    ans = re.sub(r"\\text\{([^{}]*)\}", r"\1", ans)
    ans = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", ans)
    ans = ans.strip()
    return letter_from_text(ans, options), ans[:120]


# --------------------------------------------------------------------- runner
class Runner:
    def __init__(self, args):
        from transformers import AutoTokenizer
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.model_path)
        self.budget = args.max_model_len - args.max_tokens - args.slack

    def build(self, row):
        """-> (prompt, n_input_tokens, over_budget)

        NO truncation: LongReason clues sit at random positions, so cutting
        context can delete the answer and would silently turn a capacity problem
        into a (wrong) accuracy number. Over-budget rows are flagged and scored 0.
        """
        c = self.args.contract
        is_long = bool(row["background"])
        if c == "official":
            system = OFFICIAL_SYSTEM
            user = ((OFFICIAL_LONG if is_long else OFFICIAL_SHORT)
                    .format(background=row["background"], inquiry=row["final_inquiry"]))
        else:
            system = (THINK_SYSTEM_PROMPT if c in ("think", "think_mcq")
                      else BOXED_SYSTEM_PROMPT)
            if c.endswith("_mcq"):
                stem = TRAINED_LONG_STEM if is_long else TRAINED_SHORT_STEM
                user = stem.format(background=row["background"],
                                   inquiry=row["final_inquiry"]) + TRAINED_MCQ_SENTENCE
            else:
                tmpl = TRAINED_LONG if is_long else TRAINED_SHORT
                user = tmpl.format(background=row["background"],
                                   inquiry=row["final_inquiry"])
        prompt, n = self.render(system, user,
                                prefill=(c in ("think", "think_mcq")))
        return prompt, n, n > self.budget

    def render(self, system, user, prefill):
        p = (f"<|im_start|>system\n{system}<|im_end|>\n"
             f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        if prefill:
            p += "<think>"
        return p, len(self.tok(p, add_special_tokens=False)["input_ids"])

    def gen(self, prompt, seed):
        import requests
        r = requests.post(
            f"{self.args.api_base}/completions",
            json={"model": self.args.served_name, "prompt": prompt,
                  "max_tokens": self.args.max_tokens,
                  "temperature": self.args.temperature, "top_p": self.args.top_p,
                  "n": self.args.n, "seed": seed, "stop": ["<|im_end|>"]},
            timeout=self.args.timeout)
        r.raise_for_status()
        return [c["text"] for c in r.json()["choices"]]


def official_prompt_sha1(row):
    """sha1 of the reconstructed upstream prompt, to compare against the
    prompt_sha1 that prepare_longreason.py copied from the published column."""
    tmpl = OFFICIAL_LONG if row["background"] else OFFICIAL_SHORT
    return hashlib.sha1(tmpl.format(background=row["background"],
                                    inquiry=row["final_inquiry"]).encode()).hexdigest()


def run_tier(runner, args, rows):
    recs = []

    def one(i_r):
        i, row = i_r
        prompt, n_in, over = runner.build(row)
        opts = [tuple(o) for o in row["options"]]
        if over:
            return [{"tier": row["tier"], "qidx": row["qidx"],
                     "category": row["category"], "sample": k, "acc": 0.0,
                     "pred": "", "pred_raw": "", "gold": row["answer"],
                     "n_options": row["n_options"], "fmt_ok": False,
                     "n_in_tok": n_in, "over_budget": True, "n_out_chars": 0,
                     "error": "", "raw_tail": ""} for k in range(args.n)]
        try:
            outs = runner.gen(prompt, args.seed + i)
            err = ""
        except Exception as e:
            outs, err = [""] * args.n, f"{type(e).__name__}: {e}"[:200]
        out = []
        for k, text in enumerate(outs):
            if args.contract == "official":
                pred, raw = extract_official(text, opts)
            elif args.contract.endswith("_mcq"):
                pred, raw = extract_boxed_option(
                    text, opts, require_think=(args.contract == "think_mcq"))
            else:
                pred, raw = extract_boxed(text, opts,
                                          require_think=(args.contract == "think"))
            rec = {
                "tier": row["tier"], "qidx": row["qidx"],
                "category": row["category"], "sample": k,
                "acc": float(pred == row["answer"]), "pred": pred,
                "pred_raw": raw, "gold": row["answer"],
                "n_options": row["n_options"], "fmt_ok": bool(pred),
                "n_in_tok": n_in, "over_budget": False,
                "n_out_chars": len(text), "error": err,
                "raw_tail": text[-300:],
            }
            # Keeping the whole completion is what makes a later re-score under a
            # different extractor possible at all: `raw_tail` alone cannot answer
            # "what if we accepted <answer> tags / bare letters / mid-text boxed".
            if args.save_text:
                rec["text"] = text
            out.append(rec)
        return out

    with ThreadPoolExecutor(args.workers) as ex:
        for j, rs in enumerate(ex.map(one, enumerate(rows))):
            recs.extend(rs)
            if (j + 1) % 50 == 0:
                acc = sum(x["acc"] for x in recs) / len(recs)
                print(f"[{args.tier}] {j+1}/{len(rows)} acc={acc:.4f}", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract",
                    choices=["think", "boxed", "official",
                             "think_mcq", "boxed_mcq"],
                    default="think",
                    help="base models MUST use official; *_mcq are the "
                         "v2-aligned MCQ envelope (training shard sentence + "
                         "parenthesised-letter reward rule)")
    ap.add_argument("--tier", required=True,
                    choices=["original", "expanded", "8k", "16k", "32k", "64k", "128k"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--served-name", default="m")
    ap.add_argument("--api-base", default="http://127.0.0.1:8400/v1")
    ap.add_argument("--data-dir", required=True,
                    help="output of prepare_longreason.py (expects <tier>.jsonl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-model-len", type=int, default=41984)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--slack", type=int, default=1024,
                    help="reserve; input budget = max_model_len - max_tokens - slack")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=5, help="samples per question (Avg@N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N questions (smoke only)")
    ap.add_argument("--save-text", action="store_true", default=True,
                    help="store the full completion per sample so any later "
                         "re-score under a different extractor is possible")
    ap.add_argument("--no-save-text", dest="save_text", action="store_false")
    ap.add_argument("--gzip-records", action="store_true", default=True,
                    help="write records_longreason.jsonl.gz (full texts are big)")
    ap.add_argument("--no-gzip-records", dest="gzip_records", action="store_false")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    f = Path(args.data_dir) / f"{args.tier}.jsonl"
    if not f.exists():
        raise SystemExit(f"missing slice {f} (run prepare_longreason.py --tiers {args.tier})")
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
        print(f"WARNING --limit {args.limit}: head slice is category-skewed "
              f"(rows 0-279 are all reading comprehension). Smoke tests only.")

    runner = Runner(args)
    # cheap contract proof: reconstructed official prompt vs published sha1
    sha_ok = sum(official_prompt_sha1(r) == r["prompt_sha1"] for r in rows)
    print(f"official_prompt_sha1 match: {sha_ok}/{len(rows)}", flush=True)
    if args.contract == "official" and sha_ok != len(rows):
        raise SystemExit("official template drifted from the published prompt column "
                         "-- fix OFFICIAL_* before reporting base numbers")
    # same guard on the other side: the *_mcq envelope is only "aligned" if its
    # sentence is still byte-identical to the training shard's
    if args.contract.endswith("_mcq"):
        got = hashlib.sha1(TRAINED_MCQ_SENTENCE.encode()).hexdigest()
        if got != TRAINED_MCQ_SENTENCE_SHA1:
            raise SystemExit(f"MCQ task sentence drifted from the training shard: "
                             f"{got} != {TRAINED_MCQ_SENTENCE_SHA1}")
        print(f"trained_mcq_sentence_sha1 OK: {got}", flush=True)

    t0 = time.time()
    recs = run_tier(runner, args, rows)

    by_cat = defaultdict(list)
    by_sample = defaultdict(list)
    for r in recs:
        by_cat[r["category"]].append(r["acc"])
        by_sample[r["sample"]].append(r["acc"])
    sample_means = [sum(v) / len(v) for v in by_sample.values()]
    n_opts = {r["qidx"]: r["n_options"] for r in recs}
    summary = {
        "tier": args.tier,
        "acc": round(sum(x["acc"] for x in recs) / len(recs), 4),
        "std_across_samples": round(
            statistics.stdev(sample_means) if len(sample_means) > 1 else 0.0, 4),
        "acc_by_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
        "random_baseline": round(sum(1 / v for v in n_opts.values()) / len(n_opts), 4),
        "fmt_ok_rate": round(sum(x["fmt_ok"] for x in recs) / len(recs), 4),
        "over_budget_rate": round(sum(x["over_budget"] for x in recs) / len(recs), 4),
        "error_rate": round(sum(bool(x["error"]) for x in recs) / len(recs), 4),
        "n_in_tok_p50": int(statistics.median([x["n_in_tok"] for x in recs])),
        "n_in_tok_max": max(x["n_in_tok"] for x in recs),
        "input_budget": runner.budget,
        "n": len(recs), "n_questions": len(rows),
        "official_prompt_sha1_match": f"{sha_ok}/{len(rows)}",
        "wall_s": round(time.time() - t0, 1),
        "config": {k: v for k, v in vars(args).items()},
    }
    sys_used = {"think": THINK_SYSTEM_PROMPT, "think_mcq": THINK_SYSTEM_PROMPT,
                "boxed": BOXED_SYSTEM_PROMPT, "boxed_mcq": BOXED_SYSTEM_PROMPT,
                "official": OFFICIAL_SYSTEM}[args.contract]
    summary["config"]["system_prompt_sha1"] = hashlib.sha1(sys_used.encode()).hexdigest()
    if args.contract.endswith("_mcq"):
        summary["config"]["trained_mcq_sentence_sha1"] = TRAINED_MCQ_SENTENCE_SHA1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(x, ensure_ascii=False) for x in recs)
    if args.gzip_records:
        with gzip.open(out / "records_longreason.jsonl.gz", "wt",
                       encoding="utf-8") as fh:
            fh.write(body)
    else:
        (out / "records_longreason.jsonl").write_text(body)
    (out / "summary_longreason.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "config"},
                                indent=2))
    if summary["over_budget_rate"] > 0:
        print(f"!! over_budget={summary['over_budget_rate']:.3f} -- those rows scored 0 "
              f"WITHOUT truncation (input budget {runner.budget}, max seen "
              f"{summary['n_in_tok_max']}). Raise --max-model-len or report the "
              f"tier as out-of-capacity; do NOT truncate LongReason.")


if __name__ == "__main__":
    main()
