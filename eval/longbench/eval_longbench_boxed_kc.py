#!/usr/bin/env python3
"""LongBench-v1 subset eval for the KC512 track (repro16kmixkc bs512, Qwen2.5-7B).
Derived from eval_longbench_boxed.py with 4 deltas for the KC contract:
  1. Avg@5: single request n=5 (records carry qidx/sample for verify_v1_orm.py)
  2. --top-p arg (KC rollout: t=1.0 top_p=1.0)
  3. <think> prefill appended to the ChatML generation prompt (training contract)
  4. extraction: last boxed AFTER </think> (matches loongrl_reward_repro.py gate)
Original (boxed track): Follows Experiments/longbenchv1_evaluation.md:

- Generation contract = training contract: boxed system prompt (byte-exact
  from loongrl_repro_16k_v3), bare `{context}\n\nQuestion: {question}`
  envelope, hand-rolled ChatML (== Qwen2.5 chat template for system+user).
- Sampling = training rollout: temperature 0.6, top_p 0.95, seed 0.
- Input budget 32,768 (middle truncation), generation 8,192,
  server max_model_len 41,984 + YaRN x2 (Qwen2.5 native 32k).
- Extraction: last \\boxed{...} (same regex as the training reward,
  judge_reward_boxed.py). NO fallback: no boxed answer -> pred "" -> 0.
- Rule score sub_em: normalized two-way substring containment over
  gold + aliases. Token-F1 kept as a reference column only.
- Stage 2 (ORM verify) runs separately: verify_v1_orm.py on the records.
"""
import argparse, json, re, string, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

# Byte-exact system prompts from the training parquets.
# boxed: loongrl_repro_16k_v3 (no-think batch)
BOXED_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and "
    "the Assistant solves it. The Assistant provides the final answer enclosed "
    "within \\boxed{} tags, i.e., \\boxed{answer here}. The \\boxed{} should "
    "contain only the answer itself, with no explanation or extra words."
)
# think: loongrl_repro_16k_v3_think (see datasets/rewrite_prompt_think_boxed.py)
THINK_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and "
    "the Assistant solves it. The Assistant first thinks about the reasoning "
    "process in the mind and then provides the User with the answer. The "
    "reasoning process is enclosed within <think> </think> and answer is "
    "enclosed within \\boxed{} tags, respectively, i.e., <think> reasoning "
    "process here </think> \\boxed{answer here}."
)
SYSTEM_PROMPT = BOXED_SYSTEM_PROMPT  # overridden by --contract in main()

# Training envelope for the QA rows.
#   bare     (v3 / v3think batches): {context}\n\nQuestion: {q}
#   passages (v4officialqa batch):   official LoongRL header first
V1_TEMPLATES = {
    "bare": "{context}\n\nQuestion: {question}",
    "passages": "The following are given passages.\n{context}\n\nQuestion: {question}",
}
V1_TEMPLATE = V1_TEMPLATES["bare"]  # overridden by --envelope in main()

# Same extraction as training reward judge_reward_boxed.py::extract_boxed
# (one level of brace nesting, last match, latex-wrapper stripping).
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text):
    """Last \\boxed AFTER </think> (same gate as the training reward).
    The prompt prefills <think>, so a well-formed response closes it;
    unclosed think -> pred "" -> 0, matching training."""
    if not text:
        return ""
    if "</think>" not in text:
        return ""
    text = text.rsplit("</think>", 1)[1]
    m = _BOXED.findall(text)
    if not m:
        return ""
    ans = m[-1].strip()
    ans = re.sub(r"\\text\{([^{}]*)\}", r"\1", ans)
    ans = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", ans)
    return ans.strip()


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def sub_em(pred, gts):
    p = norm(pred)
    if not p:
        return 0.0
    for gt in gts:
        g = norm(gt)
        if g and (g in p or p in g):
            return 1.0
    return 0.0


def qa_f1(pred, gts):
    best = 0.0
    for gt in gts:
        p, g = norm(pred).split(), norm(gt).split()
        common = Counter(p) & Counter(g)
        ns = sum(common.values())
        if ns == 0 or not p or not g:
            continue
        prec, rec = ns / len(p), ns / len(g)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


class Runner:
    def __init__(self, args):
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.model_path)
        # 41984 - 8192 - 1024 = 32768 -> spec input budget exactly
        self.budget = args.max_model_len - args.max_tokens - 1024

    def render(self, user):
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>"
        )

    def truncate_middle(self, user):
        ids = self.tok(user, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.budget:
            return user
        half = self.budget // 2
        return (
            self.tok.decode(ids[:half])
            + "\n...[truncated]...\n"
            + self.tok.decode(ids[-half:])
        )

    def gen(self, user, seed):
        """Return a list of n sampled completions (shared prefill via n=)."""
        user = self.truncate_middle(user)
        r = requests.post(
            f"{self.args.api_base}/completions",
            json={
                "model": self.args.served_name,
                "prompt": self.render(user),
                "max_tokens": self.args.max_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "n": self.args.n,
                "seed": seed,
                "stop": ["<|im_end|>"],
            },
            timeout=3600,
        )
        r.raise_for_status()
        return [c["text"] for c in r.json()["choices"]]


def run_v1(runner, args):
    results = []
    for tf in sorted(Path(args.v1_dir).glob("*.jsonl")):
        rows = [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        task = tf.stem

        def one(i_r):
            i, r = i_r
            user = V1_TEMPLATE.format(context=r["context"], question=r["input"])
            outs = runner.gen(user, args.seed + i)
            if len(outs) != args.n:
                raise ValueError(f"{task} question {i}: expected {args.n} generations")
            recs = []
            for k, out in enumerate(outs):
                pred = extract_boxed(out) if not out.startswith("ERROR ") else ""
                acc = sub_em(pred, r["answers"])
                f1 = qa_f1(pred, r["answers"]) if pred else 0.0
                recs.append({
                    "task": task, "qidx": i, "sample": k,
                    "question": r["input"][:500], "pred": pred[:300],
                    "gold": r["answers"], "acc": acc, "f1": f1,
                    "boxed_found": bool(pred), "n_out_chars": len(out),
                    "raw_tail": out[-300:],
                })
            return recs

        tr = []
        with ThreadPoolExecutor(args.workers) as ex:
            for j, recs in enumerate(ex.map(one, enumerate(rows))):
                tr.extend(recs)
                if (j + 1) % 50 == 0:
                    print(f"[{task}] {j+1}/{len(rows)} sub_em={sum(x['acc'] for x in tr)/len(tr):.4f}", flush=True)
        results += tr
        print(f"[{task}] DONE sub_em={sum(x['acc'] for x in tr)/len(tr):.4f} "
              f"boxed_rate={sum(x['boxed_found'] for x in tr)/len(tr):.3f} n={len(tr)}", flush=True)

    per = defaultdict(list)
    for x in results:
        per[x["task"]].append(x["acc"])
    summary = {k: sum(v) / len(v) for k, v in per.items()}
    summary["avg_sub_em"] = sum(summary.values()) / len(summary)
    summary["boxed_rate"] = sum(x["boxed_found"] for x in results) / len(results)
    summary["avg_f1_ref"] = sum(x["f1"] for x in results) / len(results)
    summary["n"] = len(results)
    return results, summary


def main():
    global SYSTEM_PROMPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", choices=["boxed", "think"], default="boxed",
                    help="training contract of the evaluated checkpoint")
    ap.add_argument("--envelope", choices=["bare", "passages"], default="bare",
                    help="user-turn envelope of the training data")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--served-name", default="boxed")
    ap.add_argument("--api-base", default="http://127.0.0.1:8400/v1")
    ap.add_argument("--v1-dir", default="/tmp/instance_storage/0727-eval/eval_data/longbench_v1_subset")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-model-len", type=int, default=41984)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=5, help="samples per question (Avg@N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    SYSTEM_PROMPT = THINK_SYSTEM_PROMPT if args.contract == "think" else BOXED_SYSTEM_PROMPT
    global V1_TEMPLATE
    V1_TEMPLATE = V1_TEMPLATES[args.envelope]

    runner = Runner(args)
    t0 = time.time()
    results, summary = run_v1(runner, args)
    summary["wall_s"] = round(time.time() - t0, 1)
    summary["config"] = {k: v for k, v in vars(args).items() if k != "v1_dir"}
    summary["config"]["system_prompt_sha1"] = __import__("hashlib").sha1(
        SYSTEM_PROMPT.encode()).hexdigest()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records_v1.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in results))
    (out / "summary_v1.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("SUMMARY", json.dumps(summary, indent=2)[:1500])


if __name__ == "__main__":
    main()
