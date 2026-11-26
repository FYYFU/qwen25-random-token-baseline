# VERBATIM copy of loongRL-repro/verl/loongrl/loongrl_reward.py (2026-08-04)
# for the RLCSD code-isolation run. Only this header was added.
# Requires env (set by the runner; also the in-file defaults):
#   REWARD_CALC_TYPE=pure_exact_match  RULER_REWARD_CALC_TYPE=format_exact_match
# Imports verl.utils.reward_score.{math_dapo,prime_math,default_compute_score}
# lazily inside functions -- all present in RLCSD third_party/verl (verified).
"""LoongRL custom rewards for upstream verl (0.8.x) via custom_reward_function.

Self-contained port of the reward logic from LoongRL's verl fork
(verl/utils/reward_score/{longcontext_qa,mathqa_choice,ruler_multi}.py),
covering every data_source in the 8K training mix:

  custom_longcontextqa_*    hotpotqa / musique / 2wiki   (boxed answer, guarded bidirectional containment)
  custom_mathqa_choice_*    mathqa multiple choice       (boxed option letter)
  custom_rulerniah_*        ruler multikey / multivalue  (ALL values must appear in the boxed answer)
  custom_math_dapo17k       dapo17k rebuilt              (dapo_boxed: hang-proof regex/float match)
  custom_math_* (other)     fork's prime_math routing    (boxed + sympy equivalence)
  everything else           delegated to verl's default_compute_score

longcontextqa/mathqa/ruler additionally require a closed </think> tag before
the answer (REWARD_REQUIRE_THINK=0 to disable); the custom_math_ path does
not, matching the fork.

Wire it up with:
  reward.custom_reward_function.path=<this file> \
  reward.custom_reward_function.name=compute_score

Calc types are still overridable via REWARD_CALC_TYPE / RULER_REWARD_CALC_TYPE,
but the *defaults here are the values the LoongRL 8K runs require*
(pure_exact_match / format_exact_match). This matters because rewards now run
inside Ray reward-loop workers, which are not guaranteed to inherit the
driver's shell environment.
"""

import os
import re
import string

# ---------------------------------------------------------------------------
# shared helpers (identical in all three original files)
# ---------------------------------------------------------------------------


def last_boxed_only_string(s):
    idx = s.rfind("\\boxed")
    if "\\boxed " in s:
        return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(s):
        if s[i] == "{":
            num_left_braces_open += 1
        if s[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else s[idx : right_brace_idx + 1]


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def normalize_text(text):
    """Normalize text by lowercasing and removing special characters."""
    text = re.sub("[,.:\"'\\[\\]\\-=\\+\\\\|!@#$%^&*();<>?/！￥…（）—\\{\\}：”“《》？]", " ", text.lower())
    text = re.sub("import\\s[a-zA-Z\\.]+(\\sas\\s[a-zA-Z\\.]+)\\n", " ", text)
    text = re.sub("\\s+", " ", text)
    return text.strip()


def is_gt_in_pred(pred, ground_truth):
    if pred is None and ground_truth is None:
        print("WARNING: Both None")
        return True
    if pred is None or ground_truth is None:
        return False
    try:
        return normalize_text(ground_truth) in normalize_text(pred)
    except Exception:
        return ground_truth in pred


def extract_answer_section(solution_str):
    """Return the text after the closed think tag, or None if malformed.

    The chat template prefills ``<think>`` in the prompt, so a well-formed
    response contains ``</think>`` followed by the final answer. Requiring it
    prevents crediting \\boxed{} that appears inside the thinking section.
    Disable via REWARD_REQUIRE_THINK=0.
    """
    if int(os.getenv("REWARD_REQUIRE_THINK", 1)):
        if "</think>" not in solution_str:
            return None
        return solution_str.split("</think>")[-1].strip()
    return solution_str


# ---------------------------------------------------------------------------
# longcontext_qa (custom_longcontextqa_*)
# ---------------------------------------------------------------------------


def _norm_for_match(text):
    """normalize_text plus SQuAD-style article removal, so bare articles
    ("the", "a") can never satisfy the pred-in-gt containment direction."""
    text = normalize_text(text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _lc_is_match(pred, ground_truth):
    """Bidirectional normalized containment (sub_em) with guards.

    - yes/no(/noanswer) answers require exact equality: substring matching
      would hit "no" inside "north", "yes" inside "yesterday", etc.
    - pred-in-gt direction requires len(pred) >= 3 so degenerate short
      predictions (single letters, empty boxed) cannot match.
    """
    if pred is None and ground_truth is None:
        print("WARNING: Both None")
        return True
    if pred is None or ground_truth is None:
        return False
    try:
        ss1 = _norm_for_match(pred)
        ss2 = _norm_for_match(ground_truth)
        binary = {"yes", "no", "noanswer"}
        if ss1 in binary or ss2 in binary:
            return ss1 == ss2
        if not ss1:
            return False
        return (ss2 in ss1) or (len(ss1) >= 3 and ss1 in ss2)
    except Exception:
        return ground_truth in pred


def _pure_exact_match_in_string(solution_str, ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if _lc_is_match(pred, truth):
                    return 1.0
        except Exception as e:
            print(f"Error encountered: {e}")
            return retval
    return retval


def _format_exact_match_in_string(solution_str, ground_truth):
    max_boxed_limit = int(os.getenv("MAX_BOXED_LIMIT", 1))
    punish_multiple_braces = int(os.getenv("PUNISH_MULTIPLE_BRACES", 1))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    max_retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            retval = 0
            if max_boxed_limit > 0 and solution_str.count("\\boxed") > max_boxed_limit:
                raise ValueError(f"Too many boxed parts: {solution_str.count(chr(92) + 'boxed')} > {max_boxed_limit}")
            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if punish_multiple_braces > 0 and (pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1):
                    raise ValueError(f"Multiple braces found in pred: {pred}")
                if _lc_is_match(pred, truth):
                    retval = 1.0
                    if len(pred) - len(truth) > int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128)):
                        retval -= 1.0
                    if len(solution_str) - (solution_str.rfind(pred) + len(pred)) > int(os.getenv("EOT_OVER_FLOW_LIMIT", 32)):
                        retval -= 1.0
                    max_retval = max(max_retval, retval)
        except Exception as e:
            print(f"Error encountered: {e}")
            return max_retval
    return max_retval


def longcontext_qa_compute_score(solution_str, ground_truth):
    solution_str = solution_str.strip()
    # answer must come after a closed think tag; extraction below only sees
    # the post-</think> section
    solution_str = extract_answer_section(solution_str)
    if solution_str is None:
        return 0.0
    calc_type = os.getenv("REWARD_CALC_TYPE", "pure_exact_match")
    if calc_type == "pure_exact_match":
        return _pure_exact_match_in_string(solution_str, ground_truth)
    if calc_type == "format_exact_match":
        return _format_exact_match_in_string(solution_str, ground_truth)
    raise ValueError(f"Unknown REWARD_CALC_TYPE: {calc_type}")


# ---------------------------------------------------------------------------
# mathqa_choice (custom_mathqa_choice_*)
# ---------------------------------------------------------------------------


def mathqa_choice_compute_score(solution_str, ground_truth):
    solution_str = solution_str.strip()
    # ground_truth arrives as a single-element list/array after the schema
    # normalization (HF datasets requires aligned features across parquets)
    if not isinstance(ground_truth, str):
        ground_truth = list(ground_truth)[0]
    # answer must come after a closed think tag; disable via REWARD_REQUIRE_THINK=0
    solution_str = extract_answer_section(solution_str)
    if solution_str is None:
        return 0
    punish_multiple_braces = int(os.getenv("MATHQA_PUNISH_MULTIPLE_BRACES", 1))
    answer_overflow_limit = int(os.getenv("MATHQA_ANSWER_OVER_FLOW_LIMIT", 32))
    max_boxed_limit = int(os.getenv("MATHQA_MAX_BOXED_LIMIT", 4))
    eot_over_flow_limit = int(os.getenv("MATHQA_EOT_OVER_FLOW_LIMIT", 16))
    retval = 0
    try:
        if max_boxed_limit > 0 and solution_str.count("\\boxed") > max_boxed_limit:
            return 0
        boxed_part = last_boxed_only_string(solution_str)
        pred = remove_boxed(boxed_part)
        if eot_over_flow_limit > 0:
            if len(solution_str) - (solution_str.rfind(pred) + len(pred)) > eot_over_flow_limit:
                retval = 0
        if punish_multiple_braces and (pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1):
            return 0
        if answer_overflow_limit > 0 and len(pred) > answer_overflow_limit:
            return 0
        pred_option = re.search(r"\((\w)\)", pred)
        if pred_option:
            if len(pred_option.groups()) > 1:
                retval = 0
            elif pred_option.group(1).lower() == ground_truth.lower():
                retval = 1.0
            else:
                retval = 0
        else:
            retval = 0
    except Exception as e:
        print(f"Error encountered: {e}")
        return 0
    return retval


# ---------------------------------------------------------------------------
# ruler multikey / multivalue (custom_rulerniah_*)
# ---------------------------------------------------------------------------


def _ruler_pure_exact_match(solution_str, ground_truth):
    # NOTE: ruler keeps the ORIGINAL (unidirectional, unguarded) matcher —
    # the fork only upgraded longcontext_qa's is_gt_in_pred.
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if is_gt_in_pred(pred, truth):
                    return 1.0
        except Exception as e:
            print(f"Error encountered: {e}")
            return retval
    return retval


def _ruler_format_exact_match(solution_str, ground_truth):
    max_boxed_limit = int(os.getenv("RULER_MAX_BOXED_LIMIT", 2))
    punish_multiple_braces = int(os.getenv("RULER_PUNISH_MULTIPLE_BRACES", 1))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    max_retval = 0
    try:
        boxed_part = last_boxed_only_string(solution_str)
        retval = 0
        if max_boxed_limit > 0 and solution_str.count("\\boxed") > max_boxed_limit:
            raise ValueError(f"Too many boxed parts: {solution_str.count(chr(92) + 'boxed')} > {max_boxed_limit}")
        if boxed_part is not None:
            pred = remove_boxed(boxed_part)
            if punish_multiple_braces > 0 and (pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1):
                raise ValueError(f"Multiple braces found in pred: {pred}")
            # different from longcontextqa: ALL ground-truth values must appear
            correct_flag = all(is_gt_in_pred(pred, truth) for truth in ground_truth)
            if correct_flag:
                retval = 1.0
                if len(pred) - sum(len(t) for t in ground_truth) > int(os.getenv("RULER_ANSWER_OVER_FLOW_LIMIT", 128)):
                    retval -= 1.0
                if len(solution_str) - (solution_str.rfind(pred) + len(pred)) > int(os.getenv("RULER_EOT_OVER_FLOW_LIMIT", 32)):
                    retval -= 1.0
                max_retval = max(max_retval, retval)
    except Exception as e:
        print(f"Error encountered: {e}")
        return max_retval
    return max_retval


def ruler_multi_compute_score(solution_str, ground_truth):
    solution_str = solution_str.strip()
    # answer must come after a closed think tag; disable via REWARD_REQUIRE_THINK=0
    solution_str = extract_answer_section(solution_str)
    if solution_str is None:
        return 0
    # NOTE: default differs from the original fork (pure_exact_match): the 8K
    # runs REQUIRE format_exact_match — pure_exact_match gives full credit for
    # matching any single value, which breaks multivalue.
    calc_type = os.getenv("RULER_REWARD_CALC_TYPE", "format_exact_match")
    if calc_type == "pure_exact_match":
        return _ruler_pure_exact_match(solution_str, ground_truth)
    if calc_type == "format_exact_match":
        return _ruler_format_exact_match(solution_str, ground_truth)
    raise ValueError(f"Unknown RULER_REWARD_CALC_TYPE: {calc_type}")


# ---------------------------------------------------------------------------
# dapo_boxed (custom_math_dapo17k) — port of the fork's dapo_boxed.py
# ---------------------------------------------------------------------------


def dapo_boxed_compute_score(solution_str, ground_truth):
    """Lightweight boxed-answer scorer for DAPO-17k (custom_math_dapo17k).

    DAPO-Math-17k answers are integers by construction, so full sympy
    equivalence (prime_math) is unnecessary — and dangerous inside Ray
    workers: its signal-based timeouts cannot interrupt sympy C-level loops,
    which can hang the reward loop until the raylet watchdog kills the
    worker. This scorer is pure regex/float comparison and cannot hang.
    """
    # helpers shared with upstream's math_dapo module
    from verl.utils.reward_score.math_dapo import (
        last_boxed_only_string as _dapo_last_boxed,
        normalize_final_answer as _dapo_normalize,
        remove_boxed as _dapo_remove_boxed,
    )

    # think-gate, consistent with longcontext_qa/mathqa_choice/ruler_multi;
    # disable via REWARD_REQUIRE_THINK=0
    if int(os.getenv("REWARD_REQUIRE_THINK", 1)):
        if "</think>" not in solution_str:
            return 0.0
        solution_str = solution_str.split("</think>")[-1]

    boxed = _dapo_last_boxed(solution_str)
    if boxed is None:
        return 0.0
    try:
        pred = _dapo_normalize(_dapo_remove_boxed(boxed)).strip()
    except Exception:
        return 0.0

    # str -> single gt; list/tuple/ndarray (post schema normalization) -> iterate
    gts = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    for gt in gts:
        gt = _dapo_normalize(str(gt)).strip()
        if pred == gt:
            return 1.0
        # numeric fallback: "1,234" / "1234.0" / " 1234" all count
        try:
            if abs(float(pred.replace(",", "")) - float(gt.replace(",", ""))) < 1e-6:
                return 1.0
        except (ValueError, OverflowError):
            pass
    return 0.0


# ---------------------------------------------------------------------------
# dispatcher (entry point for verl's custom_reward_function)
# ---------------------------------------------------------------------------


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source.startswith("custom_longcontextqa_"):
        return longcontext_qa_compute_score(solution_str, ground_truth)
    if data_source.startswith("custom_mathqa_choice_"):
        return mathqa_choice_compute_score(solution_str, ground_truth)
    if data_source.startswith("custom_rulerniah_"):
        return ruler_multi_compute_score(solution_str, ground_truth)
    if data_source in ("custom_math_dapo17k", "math_dapo"):
        # DAPO-17k answers are integers: use the hang-proof regex scorer
        # instead of prime_math (sympy can deadlock inside Ray workers).
        # "math_dapo" = RLCSD-native rename (loongrl_mix_8k_orz_rlsd build):
        # required so MATH_DATA_SOURCES exact-match recognizes math rows for
        # rlsd_skip_math; scoring is identical to the custom_ route.
        return dapo_boxed_compute_score(solution_str, ground_truth)
    if data_source == "mathqa_choice":
        # RLCSD-native rename of custom_mathqa_choice_* (same scorer)
        return mathqa_choice_compute_score(solution_str, ground_truth)
    if data_source.startswith("custom_math_"):
        # other custom_math_ sources keep the fork's prime_math routing
        # (boxed extraction + sympy equivalence), returning res[0] -> 1.0/0.0.
        # NOTE: no </think> gate here, matching the fork.
        from verl.utils.reward_score import prime_math

        return float(prime_math.compute_score(solution_str, ground_truth)[0])
    # anything else: fall back to upstream default
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(data_source, solution_str, ground_truth, extra_info)
