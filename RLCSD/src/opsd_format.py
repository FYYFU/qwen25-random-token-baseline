"""Shared OPSD-style prompt formatting and math answer grading helpers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


LEGACY_DAPO_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    '(without quotes) where $Answer is the answer to the problem.\n\n'
    'Remember to put your answer on its own line after "Answer:".'
)

BOXED_ANSWER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Variant for Base models: explicitly instruct the model to use <think> tags.
THINKING_BOXED_ANSWER_INSTRUCTION = (
    "Please first think step by step inside <think>...</think> tags, "
    "and then put your final answer within \\boxed{}."
)

# Legacy: R1-Zero-style system prompt that asked the model to wrap reasoning
# in <think>/</think>. A cold-start diagnostic on Qwen3-1.7B-Base showed the
# base model never emits </think> regardless of priming, so the structure
# could not be learned through RL alone. Base configs now share the exact
# same prompt as Instruct configs -- a single user message carrying the
# \boxed{} instruction, no system prompt, no <think> requirement. This
# constant is kept only for back-compat with any external imports.
R1_ZERO_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> tags "
    "and \\boxed{} respectively, i.e., "
    "<think> reasoning process here </think> \\boxed{answer here}."
)

# Unified lighter wording. Originally rlcsd_* used a lighter prompt while
# opsd / sdpo / rlsd / srpo used a longer "derive a fresh solution" framing,
# but the longer wording was unnecessary even for those methods, so all
# methods now share the same transition prompt. The two names are kept as
# aliases for backwards compatibility with existing imports.
TEACHER_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you understand "
    "the reasoning behind each step.\n"
)
RLCSD_TEACHER_TRANSITION_PROMPT = TEACHER_TRANSITION_PROMPT


# Named wrapper variants for the teacher prompt.  Each variant supplies a
# (framing, transition) pair; the rest of the user-message scaffolding
# (Problem header, === Reference Solution Begin/End ===, final boxed-answer
# instruction) is shared so positive and negative passes still align at the
# token level -- a hard requirement for RLCSD's CFG-style cancellation.
#
# "neutral" reproduces the current production wording verbatim, so the default
# behavior of build_teacher_messages is unchanged.  "verbose" and "terse" are
# the v1 robustness-experiment perturbations validated by the
# diagnose_prompt_robustness_dists.py probe.
TEACHER_WRAPPER_VARIANTS = {
    "neutral": {
        "framing": "Here is a reference solution to this problem:",
        "transition": TEACHER_TRANSITION_PROMPT,
    },
    "verbose": {
        "framing": (
            "Below is a fully worked-out solution by a senior mathematician "
            "known for exceptionally detailed, rigorous, and pedagogically "
            "thorough explanations that justify every algebraic step and "
            "discuss the underlying intuition:"
        ),
        "transition": (
            "\n\nNow, in the same elaborate, careful, fully-justified style "
            "-- explaining your intuition at each step, motivating every "
            "move, and double-checking each algebraic manipulation -- write "
            "your own solution.\n"
        ),
    },
    "terse": {
        "framing": (
            "A reference solution is shown below. Solutions in this "
            "collection are written in an extremely terse style: minimal "
            "prose, no restating the problem, no intuition, no verification "
            "-- just the essential computation in as few steps as possible:"
        ),
        "transition": (
            "\n\nWrite your own solution in the same terse, compact style. "
            "Be as brief as possible while still arriving at the correct "
            "result.\n"
        ),
    },
}


def normalize_wrapper_variant(variant: str) -> str:
    name = str(variant or "neutral").strip().lower()
    if name not in TEACHER_WRAPPER_VARIANTS:
        raise ValueError(
            f"Unsupported teacher_wrapper_variant={variant!r}. "
            f"Supported: {sorted(TEACHER_WRAPPER_VARIANTS)}."
        )
    return name

EVAL_DATA_SOURCES = {
    "amc23", "aime24", "aime25", "math500", "amo-bench", "minerva", "hmmt25",
    "kk_3to7_test", "kk_4to8_test", "kk_8", "kk_9", "kk_10", "kk_11",
}

KK_DATA_SOURCES = {
    "kk_3to7", "kk_3to7_test",
    "kk_4to8", "kk_4to8_test",
    "kk_8", "kk_9", "kk_10", "kk_11",
}

_SIMPLE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DEGREE_SUFFIX_RE = re.compile(r"(?:\^\{?\\circ\}?|\\degree|\\deg|\u00b0)$")
_MATH_BOXED_SUFFIX_RE = re.compile(
    r"\n+\s*(?:Choose the correct option\.\s*)?"
    r"Put (?:your final answer|the letter of your answer) in \\boxed\{\}\.\s*$",
    re.IGNORECASE)


def strip_legacy_math_prompt(text: str) -> str:
    """Strip legacy/local prompt wrappers and keep only the raw problem text."""
    if text is None:
        return ""

    text = str(text).strip()
    legacy_prefix = f"{LEGACY_DAPO_INSTRUCTION}\n\n"
    if text.startswith(legacy_prefix):
        return text[len(legacy_prefix):].strip()

    train_prefix = "Problem: "
    for instruction in (BOXED_ANSWER_INSTRUCTION, THINKING_BOXED_ANSWER_INSTRUCTION):
        suffix = f"\n\n{instruction}"
        if text.startswith(train_prefix) and text.endswith(suffix):
            return text[len(train_prefix):-len(suffix)].strip()
        if text.endswith(suffix):
            return text[:-len(suffix)].strip()

    # The long-context build stores math rows with dataset-specific boxed
    # suffixes. Remove them before rebuilding the math prompt so options and
    # the actual problem remain, without duplicating answer instructions.
    return _MATH_BOXED_SUFFIX_RE.sub("", text).strip()


def _answer_instruction(thinking: bool = False) -> str:
    return THINKING_BOXED_ANSWER_INSTRUCTION if thinking else BOXED_ANSWER_INSTRUCTION


def build_train_rollout_prompt(problem: str, thinking: bool = False) -> str:
    return f"Problem: {strip_legacy_math_prompt(problem)}\n\n{_answer_instruction(thinking)}"


def build_eval_rollout_prompt(problem: str, thinking: bool = False) -> str:
    return f"{strip_legacy_math_prompt(problem)}\n\n{_answer_instruction(thinking)}"


def build_rollout_prompt(problem: str, data_source: Optional[str] = None, thinking: bool = False) -> str:
    problem = strip_legacy_math_prompt(problem)
    if str(data_source or "").lower() in EVAL_DATA_SOURCES:
        return build_eval_rollout_prompt(problem, thinking=thinking)
    return build_train_rollout_prompt(problem, thinking=thinking)


def build_rollout_messages(problem: str, data_source: Optional[str] = None, thinking: Optional[bool] = None) -> list[dict]:
    # `thinking` was used to fork between an R1-Zero-style Base prompt
    # (system + <think> requirement) and the plain Instruct prompt. The Base
    # branch was retired (see module docstring on R1_ZERO_SYSTEM_PROMPT), so
    # both Base and Instruct now produce the same single-user-message prompt.
    # The argument is accepted but ignored.
    del thinking
    return [{"role": "user", "content": build_rollout_prompt(problem, data_source=data_source)}]


# Tail format reminders appended to the END of the user message (2026-08-11
# tailfmt data contract; probe: LBv1 OOD format 79.4%->99.4%). When the
# STUDENT rollout prompt carries a tail, every teacher/evidence view must
# carry the SAME tail, otherwise delta_t / gamma_t mix "evidence changed"
# with "instruction changed" (the tail shifts behavior by 20 points, so the
# contamination is material). Threaded as a PRESET NAME (single token) so it
# survives the hydra CLI; resolved here.
VIEW_PROMPT_TAILS = {
    "none": "",
    "boxed_v1": (
        "Provide your final answer enclosed in \\boxed{}, with only the "
        "answer itself inside the box."
    ),
}


def resolve_view_prompt_tail(preset: Optional[str]) -> str:
    key = str(preset or "none").strip().lower()
    if key not in VIEW_PROMPT_TAILS:
        raise ValueError(
            f"unknown view_prompt_tail_preset {preset!r}; known: {sorted(VIEW_PROMPT_TAILS)}"
        )
    return VIEW_PROMPT_TAILS[key]


def _append_view_tail(user_content: str, view_tail: str) -> str:
    return f"{user_content}\n\n{view_tail}" if view_tail else user_content


def build_evidence_removed_messages(
    question: str,
    minusgold_context: str,
    student_system_prompt: Optional[str] = None,
    include_document_header: bool = True,
    view_tail: str = "",
    envelope_header: Optional[str] = None,
) -> list[dict]:
    """Build the long-context policy view with annotated gold evidence removed.

    The envelope must match the student prompt exactly apart from the removed
    gold evidence, otherwise gamma_t = log p(y_t | full) - log p(y_t | minusgold)
    mixes "gold evidence removed" with "prompt envelope changed". Most
    long-context datasets wrap the passage as
    ``Document:\\n<context>\\n\\nQuestion: <q>`` (header fixed 2026-07-26; it
    was previously omitted), but merged datasets such as
    ``mix_longrlvr_loongrlqahalf_16k`` contain rows WITHOUT the header, so
    the caller must pass ``include_document_header`` per row, matching the
    student's actual envelope (added 2026-07-29).
    """
    question = str(question or "").strip()
    minusgold_context = str(minusgold_context or "").strip()
    if not question:
        raise ValueError("Evidence-removed prompting requires a non-empty question")
    if not minusgold_context:
        raise ValueError("Evidence-removed prompting requires a non-empty minusgold_context")

    messages = []
    if student_system_prompt:
        messages.append({"role": "system", "content": str(student_system_prompt)})
    if envelope_header is not None:
        # 2026-08-11 fix: reuse the student's ACTUAL envelope prefix verbatim.
        # The boolean include_document_header only knew the "Document:\n" header;
        # datasets using other headers (e.g. the KC512 passages header) got a
        # bare minusgold view while the student's full view kept its header, so
        # gamma_S carried a header-removal component on every token.
        user_content = f"{envelope_header}{minusgold_context}\n\nQuestion: {question}"
    elif include_document_header:
        user_content = f"Document:\n{minusgold_context}\n\nQuestion: {question}"
    else:
        user_content = f"{minusgold_context}\n\nQuestion: {question}"
    messages.append({
        "role": "user",
        "content": _append_view_tail(user_content, view_tail),
    })
    return messages


def build_teacher_noevidence_messages(
    question: str,
    student_system_prompt: Optional[str] = None,
    view_tail: str = "",
) -> list[dict]:
    """Build the teacher view with its privileged evidence removed.

    This is the teacher-side analogue of ``build_evidence_removed_messages``.
    It must reproduce the ``mode=gold_context`` teacher envelope exactly apart
    from the privileged block, so that

        gamma_T = log p_T(y_t | gold + question) - log p_T(y_t | question)

    isolates "how much of the teacher's predictive ability at this token comes
    from the evidence it actually holds". The gold-context teacher wrapper is
    ``f"{privileged_text}\\n\\nQuestion: {question}"`` with the student system
    message reused, so dropping the privileged block leaves just the question.

    Note this deliberately has no ``Document:`` header: the gold-context
    teacher view does not have one either, and consistency *within* this pair
    is what gamma_T requires.
    """
    question = str(question or "").strip()
    if not question:
        raise ValueError(
            "Teacher no-evidence prompting requires a non-empty question"
        )

    messages = []
    if student_system_prompt:
        messages.append({"role": "system", "content": str(student_system_prompt)})
    messages.append({"role": "user",
                     "content": _append_view_tail(f"Question: {question}", view_tail)})
    return messages


def normalize_privileged_text_mode(mode: str) -> str:
    normalized = str(mode or "solution_answer").strip().lower().replace("-", "_")
    if normalized in {"solution+answer", "solution_and_answer"}:
        normalized = "solution_answer"
    if normalized in {"answer", "answer_only"}:
        normalized = "answer_only"
    if normalized in {"gold_paragraphs", "gold_context_only"}:
        normalized = "gold_context"
    if normalized not in {"solution_answer", "answer_only", "gold_context"}:
        raise ValueError(
            f"Unsupported privileged_text_mode={mode!r}. "
            "Supported values: solution_answer (alias: solution+answer), "
            "answer_only, gold_context (aliases: gold_paragraphs, gold_context_only)."
        )
    return normalized


def build_teacher_privileged_text(
    answer: str,
    solution: str,
    mode: str,
    privileged_context: Optional[str] = None,
) -> tuple[str, str]:
    mode = normalize_privileged_text_mode(mode)
    answer = str(answer).strip()
    solution = str(solution or "").strip()
    if mode == "solution_answer":
        if not solution:
            raise ValueError("privileged_text_mode=solution_answer requires a non-empty GT solution")
        return f"{solution}\n\nCorrect final answer: {answer}", mode
    if mode == "gold_context":
        context = str(privileged_context or "").strip()
        if not context:
            raise ValueError("privileged_text_mode=gold_context requires a non-empty gold context")
        # Long-context rows can provide both retrieved gold paragraphs and a
        # verified solution. Keep both in the privileged evidence block while
        # leaving the student-facing long context out of the teacher prompt.
        if solution:
            context = f"{context}\n\nSolution:\n{solution}"
        return context, mode
    return f"Correct final answer: {answer}", mode


def build_teacher_messages(
    problem: str, answer: str, solution: str, mode: str,
    transition_prompt: Optional[str] = None,
    framing_prompt: Optional[str] = None,
    wrapper_variant: str = "neutral",
    thinking: bool = False,
    privileged_context: Optional[str] = None,
    question: Optional[str] = None,
    student_system_prompt: Optional[str] = None,
    view_tail: str = "",
) -> tuple[str, list[dict], str]:
    """Build the user-side teacher message.

    `wrapper_variant` selects a named (framing, transition) pair from
    TEACHER_WRAPPER_VARIANTS.  "neutral" reproduces the original production
    wording.  `framing_prompt` and `transition_prompt` are escape hatches that
    override the variant on a per-call basis (used by the robustness probe
    script).  If both `wrapper_variant` and an override are supplied, the
    override wins.

    For ``mode=gold_context``, ``question`` and ``privileged_context`` define
    the teacher input directly; a supplied ``solution`` is appended to the
    privileged evidence block.  ``problem`` is deliberately not used as the
    teacher context in that mode, so the full student passage cannot leak into
    the privileged forward pass.  When ``student_system_prompt`` is supplied,
    the long-context system/user wrapper is reused while only the context is
    replaced.
    """
    variant = TEACHER_WRAPPER_VARIANTS[normalize_wrapper_variant(wrapper_variant)]
    if framing_prompt is None:
        framing_prompt = variant["framing"]
    if transition_prompt is None:
        transition_prompt = variant["transition"]
    problem = strip_legacy_math_prompt(problem)
    privileged_text, effective_mode = build_teacher_privileged_text(
        answer=answer,
        solution=solution,
        mode=mode,
        privileged_context=privileged_context)
    # `thinking` previously toggled <think>/</think> requirement on the Base
    # path; that path was retired (Qwen3-Base cannot bootstrap </think>). Math
    # teacher prompts keep the plain boxed-answer wrapper, while long-context
    # gold-context prompts reuse the student's system message when available.
    del thinking
    if effective_mode == "gold_context":
        teacher_question = strip_legacy_math_prompt(
            problem if question is None else str(question)
        )
        if student_system_prompt is not None:
            # Long-context datasets use a shared system message and a user
            # message containing the context followed by the question. Keep
            # that outer prompt structure while replacing the student passage
            # with privileged context for the teacher.
            user_content = _append_view_tail(
                f"{privileged_text}\n\nQuestion: {teacher_question}", view_tail)
            messages = []
            if student_system_prompt:
                messages.append({"role": "system", "content": student_system_prompt})
            messages.append({"role": "user", "content": user_content})
            return privileged_text, messages, effective_mode
        user_content = (
            f"{privileged_text}\n\n"
            f"Question: {teacher_question}\n\n"
            f"{BOXED_ANSWER_INSTRUCTION}"
        )
    else:
        user_content = (
            f"Problem: {problem}\n\n"
            f"{framing_prompt}\n"
            "=== Reference Solution Begin ===\n"
            f"{privileged_text}\n"
            "=== Reference Solution End ==="
            f"{transition_prompt}\n"
            f"{BOXED_ANSWER_INSTRUCTION}"
        )
    messages = [{"role": "user", "content": user_content}]
    return privileged_text, messages, effective_mode


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer using the official OPSD logic."""
    if text is None:
        return None

    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def _normalize_fallback_string(text: str) -> str:
    return str(text).replace("$", "").replace(" ", "").lower().strip()


def _strip_answer_wrappers(text: str) -> str:
    text = str(text).strip()
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        text = boxed
    text = text.strip().strip("$").strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    return text.replace(r"\left", "").replace(r"\right", "").strip()


def _canonical_simple_number(text: str) -> Optional[Decimal]:
    text = _strip_answer_wrappers(text)
    text = _DEGREE_SUFFIX_RE.sub("", text.replace(" ", "").replace(",", "").replace("\u2212", "-"))
    if not _SIMPLE_NUMBER_RE.fullmatch(text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _simple_numeric_answers_match(predicted: str, ground_truth: str) -> Optional[bool]:
    pred_number = _canonical_simple_number(predicted)
    gt_number = _canonical_simple_number(ground_truth)
    if pred_number is None or gt_number is None:
        return None
    return pred_number == gt_number


_KK_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z\-']*)\s*(?:=|:|is(?:\s+(?:a|an))?)\s*(knight|knave)",
    re.IGNORECASE,
)


def parse_kk_assignment(text: Optional[str]) -> dict[str, str]:
    """Parse a K&K assignment string into a {name_lower: 'knight'|'knave'} dict.

    Accepts the natural-language form ("Alice is a knight, Bob is a knave"),
    equals form ("Alice=knight"), or colon form. Strips a leading
    `\\boxed{...}` wrapper if present. First-occurrence wins per name.
    """
    if text is None:
        return {}
    text = str(text).strip()
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        text = boxed
    result: dict[str, str] = {}
    for name, role in _KK_PAIR_RE.findall(text):
        key = name.lower()
        if key not in result:
            result[key] = role.lower()
    return result


def grade_kk_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """Compare predicted vs ground-truth K&K assignments (order-insensitive)."""
    gt = parse_kk_assignment(ground_truth)
    if not gt:
        return False
    pred = parse_kk_assignment(predicted)
    return pred == gt


def grade_boxed_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """Grade a boxed answer using official OPSD math_verify logic with string fallback."""
    if predicted is None:
        return False

    numeric_match = _simple_numeric_answers_match(predicted, ground_truth)
    if numeric_match is not None:
        return numeric_match

    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ModuleNotFoundError(
            "math_verify is required for OPSD-aligned grading. Install `math-verify` in the training env."
        ) from exc

    try:
        pred_text = predicted if "$" in predicted else f"${predicted}$"
        gt_text = ground_truth if "$" in str(ground_truth) else f"${ground_truth}$"
        pred_parsed = parse(pred_text, fallback_mode="no_fallback")
        gt_parsed = parse(gt_text, fallback_mode="no_fallback")
        return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
    except Exception:
        return _normalize_fallback_string(predicted) == _normalize_fallback_string(ground_truth)
