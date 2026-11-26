"""
Entry point for self-distillation training using verl's infrastructure.

Self-distillation flow per rollout step:
1. Student rollout: vLLM generates responses (verl standard)
2. Privileged text extraction: read GT solution/answer from the dataset batch
3. Teacher forward: compute log probs with privileged context on the same response tokens
4. Actor update with distillation loss

Usage:
    python -m src.self_distill_main <verl hydra config overrides>
"""

import os
import sys
import json
import random
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "verl"))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import open_dict

from src.opsd_format import (
    build_evidence_removed_messages,
    build_rollout_messages,
    build_teacher_messages,
    build_teacher_noevidence_messages,
    resolve_view_prompt_tail,
    RLCSD_TEACHER_TRANSITION_PROMPT,
    extract_boxed_answer,
    normalize_privileged_text_mode,
    strip_legacy_math_prompt)
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.utils.chat_template import apply_chat_template as _verl_apply_chat_template

RLCSD_LOSS_MODES = {"rlcsd","rlsd_ectr", "opsd_ectr"}
EVIDENCE_GATED_LOSS_MODES = {"eg_rlsd"}
# CE methods need the gold and question-only teacher views, but no
# evidence-removed (minusgold) policy forward or minusgold_context column.
# ce_rlsd_opsd_ectr is a CE method for data purposes: it reads exactly the same
# two views, then ALSO distills onto their contrast in the top-k subspace
# (core_algos.compute_policy_loss_ce_rlsd_opsd_ectr), so it needs no extra
# column or forward beyond ce_rlsd's.
CE_CONTRAST_LOSS_MODES = {"ce_rlsd", "ce_gate_rlsd", "ce_rlsd_opsd_ectr"}
# grpo_opsd_ectr reads the SAME two teacher views, but only as the two sides of
# the distill target -- it never modulates the advantage, so it is not a CE
# method. Same batch, different consumer; keep the two sets distinct so the CE
# name keeps meaning "the advantage is modulated by a teacher contrast".
TWO_TEACHER_VIEW_LOSS_MODES = CE_CONTRAST_LOSS_MODES | {"grpo_opsd_ectr"}
TEACHER_GATE_MODES = ("off", "diagnose", "gate")
# teacher_view selects WHICH privileged context the teacher sees. Validated
# rather than compared bare: every site below is a `== "..."` string test, so a
# typo would silently fall through to gold_only (short gold paragraphs, the
# pre-ptr teacher) and the arm would train against the wrong teacher with no
# error anywhere -- the same silent-fallback shape as teacher_mode=fixed under
# full fine-tuning.
#   gold_only      : gold paragraphs only (pre-2026-08-18 behaviour)
#   ptr_gold       : full document + pointer -> gold. ONE view; nothing reads
#                    ptr_random. This is what the distill-only arms want.
#   ptr_contrastive: ptr_gold, and ALSO stash ptr_random for the CE second view.
#                    Only the CE/eg branches build that view.
#   ptr_placebo    : mirror of ptr_contrastive with the pointer aimed at a RANDOM
#                    passage -- the direction control.
TEACHER_VIEWS = ("gold_only", "ptr_gold", "ptr_contrastive", "ptr_placebo",
                 "gold_hints")
PTR_TEACHER_VIEWS = ("ptr_gold", "ptr_contrastive", "ptr_placebo")


def _resolve_teacher_view(self) -> str:
    view = str(_custom_cfg_get(self, "teacher_view", "gold_only") or "gold_only").strip()
    if view not in TEACHER_VIEWS:
        raise ValueError(
            f"teacher_view must be one of {TEACHER_VIEWS}, got {view!r}. A typo here "
            "would silently degrade to gold_only and train the wrong teacher."
        )
    return view


def _normalize_teacher_gate_mode(mode) -> str:
    """Validate teacher_gate_mode; ``off`` keeps the batch exactly as before."""
    normalized = str(mode or "off").strip().lower().replace("-", "_")
    if normalized in {"none", "false", "0", ""}:
        normalized = "off"
    if normalized in {"true", "1", "on"}:
        normalized = "gate"
    if normalized not in TEACHER_GATE_MODES:
        raise ValueError(
            f"teacher_gate_mode must be one of {TEACHER_GATE_MODES}, got {mode!r}"
        )
    return normalized

# rlcsd_5 averages teacher_wrong_lp over up to K non-self negative siblings.
# 4 covers the typical group_size=8 case (3-7 non-self negatives) without
# ballooning per-rollout teacher-forward cost.
RLCSD_DEFAULT_K_MAX = 4
DISTILL_LOSS_MODES = {
    "opsd",
    # grpo_opsd keeps the GRPO policy gradient and ADDS the OPSD distill term
    # (core_algos.compute_policy_loss_grpo_opsd); it needs the same privileged
    # gold-context teacher view as opsd.
    "grpo_opsd",
    "sdpo",
    "rlsd",
    "srpo",
    *EVIDENCE_GATED_LOSS_MODES,
    *TWO_TEACHER_VIEW_LOSS_MODES,
    *RLCSD_LOSS_MODES,
}
MATH_DATA_SOURCES = {
    "math_dapo",
    "mathqa_choice",
    "math_deepmath",
    "dapo_math_17k",
    "openthoughts_114k_math_filtered",
    "deepmath_filtered",
    "deepmath_filtered_level5_7",
    "deepmath_filtered_level6_8",
    "deepmath_filtered_level7_10",
    "amc23",
    "aime24",
    "aime25",
}
LONG_CONTEXT_QA_ABILITY = "longcontext_qa"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for segment in content:
            if isinstance(segment, dict):
                if segment.get("type") == "text":
                    parts.append(str(segment.get("text", "")))
                elif "content" in segment:
                    parts.append(str(segment.get("content", "")))
            elif segment is not None:
                parts.append(str(segment))
        return " ".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _raw_prompt_to_problem(raw_prompt):
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, list):
        user_parts = []
        fallback_parts = []
        for msg in raw_prompt:
            if not isinstance(msg, dict):
                continue
            text = _content_to_text(msg.get("content", ""))
            if not text:
                continue
            fallback_parts.append(text)
            if msg.get("role") == "user":
                user_parts.append(text)
        parts = user_parts or fallback_parts
        return strip_legacy_math_prompt("\n".join(parts).strip())

    if raw_prompt is None:
        return ""
    return strip_legacy_math_prompt(str(raw_prompt).strip())


def _raw_prompt_to_system_prompt(raw_prompt):
    """Return the stored system message, preserving None when unavailable."""
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    if not isinstance(raw_prompt, list):
        return None
    for message in raw_prompt:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        return _content_to_text(message.get("content", "")).strip()
    return ""


def _student_system_prompts(raw_prompts, batch_size):
    if raw_prompts is None:
        return None
    raw_prompts = _collapse_repeated_to_batch(raw_prompts, batch_size)
    return [_raw_prompt_to_system_prompt(raw_prompt) for raw_prompt in raw_prompts]


def _raw_prompt_has_document_header(raw_prompt) -> bool:
    """True when the student's user message wraps the passage as ``Document:``.

    Merged datasets (e.g. mix_longrlvr_loongrlqahalf_16k) contain rows both
    with and without the header. The evidence-removed view must mirror the
    student's actual envelope per row, otherwise gamma_S measures
    "envelope changed" on the headerless rows. Defaults to True (the majority
    envelope) when the raw prompt is unavailable.
    """
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    if not isinstance(raw_prompt, list):
        return True
    for message in raw_prompt:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        return _content_to_text(message.get("content", "")).startswith("Document:\n")
    return True


# Known long-context envelope headers, most specific first. Extend when a new
# dataset introduces a new envelope (the assert in _raw_prompt_envelope_header
# will catch unknown ones on contract-bearing rows).
_KNOWN_ENVELOPE_HEADERS = (
    "The following are given passages.\n",
    "Document:\n",
)


def _raw_prompt_envelope_header(raw_prompt) -> str:
    """Return the student's user-message envelope prefix VERBATIM ("" if bare).

    2026-08-11 fix: the boolean _raw_prompt_has_document_header only knew the
    ``Document:`` header, so datasets with other headers (e.g. the KC512
    ``The following are given passages.`` header) rendered the minusgold view
    bare while the student's full view kept its header -- gamma_S then carried
    a header-removal component on every token. Reusing the actual prefix keeps
    the evidence-removed view byte-aligned with the student prompt apart from
    the removed gold. Falls back to "Document:\n" (legacy default) when the
    raw prompt is unavailable.
    """
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    if not isinstance(raw_prompt, list):
        return "Document:\n"
    for message in raw_prompt:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = _content_to_text(message.get("content", ""))
        for header in _KNOWN_ENVELOPE_HEADERS:
            if content.startswith(header):
                return header
        return ""
    return "Document:\n"


def _policy_loss_cfg(self):
    return self.config.actor_rollout_ref.actor.policy_loss


def _rollout_custom_cfg(self):
    rollout_cfg = self.config.actor_rollout_ref.get("rollout", {})
    custom_cfg = rollout_cfg.get("custom", None)
    return custom_cfg if custom_cfg is not None else {}


def _cfg_get(self, key, default):
    return _policy_loss_cfg(self).get(key, default)


def _custom_cfg_get(self, key, default):
    custom_cfg = _rollout_custom_cfg(self)
    if key in custom_cfg:
        return custom_cfg.get(key)
    return _cfg_get(self, key, default)


def _cfg_bool(self, key, default=False):
    value = _cfg_get(self, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _custom_cfg_bool(self, key, default=False):
    value = _custom_cfg_get(self, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _teacher_total_token_budget(self) -> int:
    actor_cfg = self.config.actor_rollout_ref.actor
    budget = actor_cfg.get("ppo_max_token_len_per_gpu", None)
    if budget is not None:
        return int(budget)
    return int(self.config.data.max_prompt_length) + int(self.config.data.max_response_length)


def _teacher_chat_template_kwargs(self):
    kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
    kwargs["enable_thinking"] = _custom_cfg_bool(self, "teacher_enable_thinking", True)
    return kwargs


def _student_chat_template_kwargs(self):
    kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
    kwargs.pop("tokenize", None)
    kwargs.pop("return_dict", None)
    kwargs.pop("return_tensors", None)
    return kwargs


def _render_chat(self, messages, **kwargs):
    """Render via verl's apply_chat_template wrapper (NOT the raw HF
    tokenizer): the wrapper honors force_think_prefix so teacher/student
    prompts stay token-aligned with the rollout prompts when the <think>
    prefill contract is enabled. The raw jinja template would silently
    swallow the kwarg."""
    return _verl_apply_chat_template(
        self.tokenizer, messages, tokenize=False,
        add_generation_prompt=True, **kwargs)


def _compute_prefilter_metric_overrides(self, batch: DataProto) -> dict[str, float]:
    """Capture trainer-side metrics that should stay on the pre-filter rollout batch."""
    required_keys = {"token_level_scores", "token_level_rewards", "advantages", "returns", "response_mask"}
    if any(key not in batch.batch.keys() for key in required_keys):
        return {}
    prefilter_metrics = compute_data_metrics(batch=batch, use_critic=getattr(self, "use_critic", True))
    kept_prefixes = ("critic/", "response_length/", "response_length_non_aborted/")
    kept_exact_keys = {"response/aborted_ratio"}
    return {
        key: value
        for key, value in prefilter_metrics.items()
        if key.startswith(kept_prefixes) or key in kept_exact_keys
    }


def _get_actor_dp_size(self) -> int:
    """Best-effort lookup for actor data-parallel size during training."""
    try:
        if hasattr(self, "_get_dp_size") and getattr(self, "actor_rollout_wg", None) is not None:
            return int(self._get_dp_size(self.actor_rollout_wg, "actor"))
    except Exception:
        pass
    return 1


def _prepare_actor_update_batch(self, batch: DataProto) -> DataProto:
    """Pad rlcsd actor batches to DP divisibility with zero-loss replicas."""
    if _cfg_get(self, "loss_mode", "vanilla") not in RLCSD_LOSS_MODES:
        return batch

    batch_size = len(batch)
    actor_dp_size = _get_actor_dp_size(self)
    if batch_size == 0 or actor_dp_size <= 1 or batch_size % actor_dp_size == 0:
        metrics = getattr(self, "_rlcsd_step_metrics", {})
        metrics["rlcsd/actor_pad_count"] = 0.0
        metrics["rlcsd/actor_padded_batch_size"] = float(batch_size)
        self._rlcsd_step_metrics = metrics
        return batch

    pad_size = actor_dp_size - (batch_size % actor_dp_size)
    pad_indices = [idx % batch_size for idx in range(pad_size)]
    padded_tail = batch.select_idxs(pad_indices)
    padded_tail.meta_info = dict(batch.meta_info)

    if "response_mask" in padded_tail.batch.keys():
        padded_tail.batch["response_mask"] = torch.zeros_like(padded_tail.batch["response_mask"])
    if "uid" in padded_tail.non_tensor_batch:
        padded_tail.non_tensor_batch["uid"] = np.asarray(
            [f"__rlcsd_pad__{i}" for i in range(pad_size)],
            dtype=object)
    metrics = getattr(self, "_rlcsd_step_metrics", {})
    metrics["rlcsd/actor_pad_count"] = float(pad_size)
    metrics["rlcsd/actor_padded_batch_size"] = float(batch_size + pad_size)
    self._rlcsd_step_metrics = metrics
    actor_batch = DataProto.concat([batch, padded_tail])
    actor_batch.meta_info = dict(batch.meta_info)
    if "global_token_num" in batch.meta_info:
        original_token_num = list(batch.meta_info.get("global_token_num", []))
        actor_batch.meta_info["global_token_num"] = original_token_num + [original_token_num[i] for i in pad_indices]
    return actor_batch


def _collapse_repeated_to_batch(values, batch_size):
    if values is None:
        return None

    values = list(values)
    if len(values) == batch_size:
        return values
    if batch_size <= 0 or len(values) % batch_size != 0:
        raise AssertionError(f"Cannot align {len(values)} items to batch size {batch_size}")

    repeat_factor = len(values) // batch_size
    return [values[i * repeat_factor] for i in range(batch_size)]


def _extract_key_ablation_contexts(self, batch, gold_contexts, data_sources):
    """Return strict per-row ablated contexts and the QA eligibility mask."""
    batch_size = len(batch)
    if len(gold_contexts) != batch_size or len(data_sources) != batch_size:
        raise AssertionError("key-ablation source fields are not batch-aligned")

    extra_infos = batch.non_tensor_batch.get("extra_info")
    if extra_infos is None:
        raise ValueError(
            "distill_key_ablation_gate=True requires extra_info in the rollout batch."
        )
    extra_infos = _collapse_repeated_to_batch(extra_infos, batch_size)

    ability_values = batch.non_tensor_batch.get("ability")
    if ability_values is not None:
        ability_values = _collapse_repeated_to_batch(ability_values, batch_size)

    tag = str(_cfg_get(
        self, "distill_key_ablation_field_tag", "gptoss120b_v1"
    ) or "").strip()
    if not tag or any(not (char.isalnum() or char == "_") for char in tag):
        raise ValueError(
            "distill_key_ablation_field_tag must contain only letters, digits, "
            f"and underscores; got {tag!r}."
        )
    text_field = f"key_evidence_ablation__{tag}"
    valid_field = f"key_evidence_ablation_valid__{tag}"
    sha_field = f"key_evidence_ablation_source_sha256__{tag}"

    contexts = []
    eligible = []
    for index, (extra_info, gold_context, data_source) in enumerate(
        zip(extra_infos, gold_contexts, data_sources)
    ):
        if ability_values is not None:
            is_qa = (
                str(ability_values[index] or "").strip().lower()
                == LONG_CONTEXT_QA_ABILITY
            )
        else:
            # Compatibility fallback for batches created by older dataset
            # collators that dropped the top-level ability column.
            is_qa = str(data_source).startswith("custom_longcontextqa")

        if not is_qa:
            # Every rank still executes the same extra teacher forward. Reusing
            # the gold prompt for ineligible rows provides a valid dummy input;
            # the explicit eligibility mask removes it from the loss.
            contexts.append(gold_context)
            eligible.append(False)
            continue

        if not isinstance(extra_info, dict):
            raise ValueError(
                f"QA row {index} has non-dict extra_info; cannot read {text_field}."
            )
        valid_value = extra_info.get(valid_field)
        is_valid = isinstance(valid_value, (bool, np.bool_)) and bool(valid_value)
        ablated_context = _content_to_text(extra_info.get(text_field, "")).strip()
        source_sha = _content_to_text(extra_info.get(sha_field, "")).strip()
        has_source_sha = re.fullmatch(r"[0-9a-f]{64}", source_sha) is not None
        if not is_valid and not ablated_context and has_source_sha:
            # An explicit source-limited QA row remains in GRPO. The populated
            # source hash distinguishes it from a malformed/missing ablation.
            contexts.append(gold_context)
            eligible.append(False)
            continue
        if not is_valid or not ablated_context or not has_source_sha:
            raise ValueError(
                f"QA row {index} is missing a validated key ablation: "
                f"{valid_field}={valid_value!r}, {text_field} length="
                f"{len(ablated_context)}, {sha_field}={source_sha!r}."
            )
        if ablated_context == str(gold_context).strip():
            raise ValueError(
                f"QA row {index} key ablation is identical to its gold context."
            )
        contexts.append(ablated_context)
        eligible.append(True)

    return contexts, eligible


def _extract_problems_answers_solutions(
    self,
    batch: DataProto,
    raw_prompts=None,
    reward_models=None,
    extra_infos=None,
    include_long_context=False,
    include_evidence_ablation=False,
):
    """Extract rollout problems and privileged fields from the dataset batch.

    The default return value remains ``(problems, answers, solutions)`` for
    the existing RLCSD path.  Long-context OPSD can opt into the two extra
    lists ``(questions, gold_contexts, data_sources)``; these are read from
    the dataset batch rather than reconstructed from the truncated rollout
    prompt. Evidence-gated training can additionally request
    ``minusgold_contexts``.
    """
    batch_size = len(batch)
    if raw_prompts is None:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    if reward_models is None:
        reward_models = batch.non_tensor_batch.get("reward_model")
    if extra_infos is None:
        extra_infos = batch.non_tensor_batch.get("extra_info")
    data_source_values = batch.non_tensor_batch.get("data_source")
    top_level_solutions = batch.non_tensor_batch.get("solution")

    raw_prompts = _collapse_repeated_to_batch(raw_prompts, batch_size) if raw_prompts is not None else None
    reward_models = _collapse_repeated_to_batch(reward_models, batch_size) if reward_models is not None else None
    extra_infos = _collapse_repeated_to_batch(extra_infos, batch_size) if extra_infos is not None else None
    data_source_values = (
        _collapse_repeated_to_batch(data_source_values, batch_size)
        if data_source_values is not None else None
    )
    top_level_solutions = (
        _collapse_repeated_to_batch(top_level_solutions, batch_size)
        if top_level_solutions is not None else None
    )

    problems = []
    ground_truths = []
    solutions = []
    questions = []
    gold_contexts = []
    ptr_random_contexts = []  # ptr_contrastive teacher (2026-08-18)
    minusgold_contexts = []
    data_sources = []
    for i in range(batch_size):
        raw_prompt = raw_prompts[i] if raw_prompts is not None else None
        reward_model = reward_models[i] if reward_models is not None else {}
        answer = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        ground_truths.append(str(answer))

        extra_info = extra_infos[i] if extra_infos is not None else {}
        data_source = data_source_values[i] if data_source_values is not None else ""
        data_source = str(data_source or "").strip().lower()
        problem = _content_to_text(extra_info.get("problem", "")) if isinstance(extra_info, dict) else ""
        if not problem:
            problem = _raw_prompt_to_problem(raw_prompt)
        if not problem and "prompts" in batch.batch:
            problem = strip_legacy_math_prompt(
                self.tokenizer.decode(batch.batch["prompts"][i], skip_special_tokens=True).strip()
            )
        problems.append(problem)

        solution = extra_info.get("solution", "") if isinstance(extra_info, dict) else ""
        if not solution and top_level_solutions is not None:
            solution = top_level_solutions[i]
        # The long-context builder stores DeepMath's worked solution in
        # cs_gold_context. Use it as the RLCSD math solution when the
        # dedicated solution field is absent.
        if (
            not solution
            and data_source in MATH_DATA_SOURCES
            and isinstance(extra_info, dict)
        ):
            solution = extra_info.get("cs_gold_context", "")
        solutions.append(_content_to_text(solution).strip())

        if isinstance(extra_info, dict):
            question = _content_to_text(extra_info.get("question", "")).strip()
            gold_context = extra_info.get("cs_gold_context", "")
            if not gold_context:
                gold_context = extra_info.get("gold_context", "")
            if not gold_context:
                gold_context = extra_info.get("gold_paragraphs", "")
            gold_context = _content_to_text(gold_context).strip()
            minusgold_context = _content_to_text(extra_info.get("minusgold_context", "")).strip()
        else:
            question = ""
            gold_context = ""
            minusgold_context = ""
        # ptr_contrastive teacher: swap gold view to full-doc+pointer->gold
        # and stash full-doc+pointer->random for the CE contrast side.
        # flag off / fields absent (ruler, math) -> unchanged behaviour.
        ptr_random_ctx = ""
        _tv = _resolve_teacher_view(self)
        if isinstance(extra_info, dict) and _tv in PTR_TEACHER_VIEWS:
            ptr_g = _content_to_text(extra_info.get("ptr_gold_context", "")).strip()
            ptr_r = _content_to_text(extra_info.get("ptr_random_context", "")).strip()
            if ptr_g and ptr_r:
                # ptr_placebo (2026-08-20) MIRRORS ptr_contrastive: the privileged
                # view points at a RANDOM passage instead of gold. Same envelope,
                # same token count, same pointer syntax -- the only thing that
                # changes is whether the pointer is right. It is the direction
                # control for any ptr arm; on the advantage channel the 20260819
                # B-vs-C pair already came out B~=C, i.e. direction was placebo
                # there, so an arm that gains here must beat this to mean
                # "grounding" rather than "perturbation".
                gold_context = ptr_r if _tv == "ptr_placebo" else ptr_g
                # ptr_gold is single-view on purpose: leaving the stash empty
                # means an accidental CE run with this flag falls back to the
                # question-only view instead of silently getting ptr_random.
                if _tv != "ptr_gold":
                    ptr_random_ctx = ptr_g if _tv == "ptr_placebo" else ptr_r
        # gold_hints composite teacher (2026-08-25): retrieval + reasoning
        # privilege in one view. gold passages keep the ORIGINAL numbering of
        # the student's full document (cs_gold_context stores them verbatim);
        # hints are the 72B bullets from the v3bullets dataset. Same fallback
        # convention as the ptr views above: rows without the column (ruler)
        # keep plain gold_only, and math rows carry no distill gradient anyway
        # (rlsd_skip_math). Downstream rendering is untouched.
        if isinstance(extra_info, dict) and _tv == "gold_hints":
            _gh_tag = str(_custom_cfg_get(
                self, "bullets_generator_tag", "qwen25_72b")).strip()
            _gh_hints = _content_to_text(
                extra_info.get(f"bullets_ans__{_gh_tag}", "")).strip()
            if _gh_hints and gold_context:
                gold_context = (
                    "The following are the gold passages from the document.\n"
                    + gold_context
                    + "\n\nReasoning hints:\n"
                    + _gh_hints
                )
                self._gold_hints_applied = True
        questions.append(question)
        gold_contexts.append(gold_context)
        minusgold_contexts.append(minusgold_context)
        data_sources.append(data_source)
        ptr_random_contexts.append(ptr_random_ctx)

    self._ptr_random_contexts = ptr_random_contexts
    # one-time guard (2026-08-25): teacher_view=gold_hints with a typo'd
    # bullets_generator_tag, or a dataset without the column, would silently
    # degrade every row to plain gold_only -- i.e. train a different teacher
    # than the one on the label. Every training batch of the 16kkc mix carries
    # LCQA rows, so the very first batch must have applied hints at least once.
    if (_resolve_teacher_view(self) == "gold_hints"
            and not getattr(self, "_gold_hints_guard_done", False)):
        self._gold_hints_guard_done = True
        if not getattr(self, "_gold_hints_applied", False):
            raise ValueError(
                "teacher_view=gold_hints was requested but no row in the first "
                "batch carried a non-empty bullets_ans column for "
                "bullets_generator_tag="
                f"{_custom_cfg_get(self, 'bullets_generator_tag', 'qwen25_72b')!r}; "
                "check the dataset (v3bullets) and the tag spelling."
            )
    if include_evidence_ablation:
        return (
            problems,
            ground_truths,
            solutions,
            questions,
            gold_contexts,
            minusgold_contexts,
            data_sources,
        )
    if include_long_context:
        return problems, ground_truths, solutions, questions, gold_contexts, data_sources
    return problems, ground_truths, solutions


def _build_teacher_prompt(problem, answer, solution, mode, transition_prompt=None,
                          wrapper_variant="neutral", thinking=False):
    privileged_text, messages, _ = build_teacher_messages(
        problem=problem,
        answer=answer,
        solution=solution,
        mode=mode,
        transition_prompt=transition_prompt,
        wrapper_variant=wrapper_variant,
        thinking=thinking)
    return privileged_text, messages


def _materialize_teacher_context(self, problem, answer, solution):
    """Build a teacher context for the rlcsd_* methods.

    rlcsd_* uses the lighter RLCSD_TEACHER_TRANSITION_PROMPT (does not ask
    the model to derive a separate answer). opsd/sdpo/rlsd/srpo continue
    to go through _build_teacher_context_payloads and keep the original
    TEACHER_TRANSITION_PROMPT.

    The "neutral" wrapper variant reproduces this lighter wording.  Setting
    teacher_wrapper_variant=verbose / terse in the YAML config swaps both
    the framing line and the transition prompt for the robustness probe.
    """
    use_thinking = _custom_cfg_bool(self, "thinking_system_prompt", False)
    wrapper_variant = _custom_cfg_get(self, "teacher_wrapper_variant", "neutral")
    privileged_text, messages = _build_teacher_prompt(
        problem=problem,
        answer=answer,
        solution=solution,
        mode="solution_answer",
        wrapper_variant=wrapper_variant,
        thinking=use_thinking)
    prompt = _render_chat(self, messages, **_teacher_chat_template_kwargs(self))
    return {
        "prompt": prompt,
        "text": privileged_text,
        "answer": answer}


def _sample_other_rollout(indices, current_idx):
    candidate_indices = [idx for idx in indices if idx != current_idx]
    if not candidate_indices:
        return None
    return random.choice(candidate_indices)


def _expand_to_rollout(values, batch_size, rollout_n):
    num_values = len(values)
    if num_values == batch_size:
        return values
    if num_values * rollout_n != batch_size:
        raise AssertionError(f"Mismatch: cached={num_values}, rollout_n={rollout_n}, batch={batch_size}")

    expanded = []
    for value in values:
        for _ in range(rollout_n):
            expanded.append(value)
    return expanded


def _build_teacher_context_payloads(
    self,
    problems,
    ground_truths,
    solutions,
    questions=None,
    gold_contexts=None,
    data_sources=None,
    student_system_prompts=None,
):
    requested_mode = normalize_privileged_text_mode(_custom_cfg_get(self, "privileged_text_mode", "solution_answer"))
    math_mode = normalize_privileged_text_mode(
        _custom_cfg_get(self, "math_privileged_text_mode", "solution_answer"))
    privileged_texts = []
    effective_modes = []
    teacher_prompts = []

    if requested_mode == "solution_answer":
        missing_indices = [i for i, solution in enumerate(solutions) if not solution]
        if missing_indices:
            preview = ", ".join(str(i) for i in missing_indices[:8])
            if len(missing_indices) > 8:
                preview = f"{preview}, ..."
            raise ValueError(
                "privileged_text_mode=solution_answer requires every sample to have a non-empty GT solution, "
                f"but {len(missing_indices)}/{len(solutions)} samples in the current batch are missing it "
                f"(batch indices: {preview})."
            )

    if questions is None:
        questions = [""] * len(problems)
    if gold_contexts is None:
        gold_contexts = [""] * len(problems)
    if data_sources is None:
        data_sources = [""] * len(problems)
    fallback_mode = "error"
    if requested_mode == "gold_context":
        fallback_spec = str(_custom_cfg_get(self, "gold_context_fallback_mode", "error")).strip().lower()
        if fallback_spec in {"", "error", "raise", "strict"}:
            fallback_mode = "error"
        else:
            fallback_mode = normalize_privileged_text_mode(fallback_spec)
            if fallback_mode == "gold_context":
                raise ValueError(
                    "gold_context_fallback_mode cannot be gold_context; use error or answer_only."
                )
        missing_indices = [
            i for i, (question, gold_context) in enumerate(zip(questions, gold_contexts))
            if data_sources[i] not in MATH_DATA_SOURCES
            and (not str(question).strip() or not str(gold_context).strip())
        ]
        if missing_indices and fallback_mode == "error":
            preview = ", ".join(str(i) for i in missing_indices[:8])
            if len(missing_indices) > 8:
                preview = f"{preview}, ..."
            raise ValueError(
                "privileged_text_mode=gold_context requires non-empty extra_info.question and "
                "extra_info.cs_gold_context for every sample, "
                f"but {len(missing_indices)}/{len(problems)} samples are missing one or both "
                f"fields (batch indices: {preview})."
            )

    math_missing_solution_indices = [
        i for i, (source, solution) in enumerate(zip(data_sources, solutions))
        if source in MATH_DATA_SOURCES and math_mode == "solution_answer" and not str(solution).strip()
    ]
    if math_missing_solution_indices:
        preview = ", ".join(str(i) for i in math_missing_solution_indices[:8])
        if len(math_missing_solution_indices) > 8:
            preview = f"{preview}, ..."
        raise ValueError(
            "math_privileged_text_mode=solution_answer requires a non-empty GT solution for math samples, "
            f"but {len(math_missing_solution_indices)} samples are missing it "
            f"(batch indices: {preview})."
        )

    use_thinking = _custom_cfg_bool(self, "thinking_system_prompt", False)
    wrapper_variant = _custom_cfg_get(self, "teacher_wrapper_variant", "neutral")
    for idx, (problem, answer, solution) in enumerate(zip(problems, ground_truths, solutions)):
        is_math = data_sources[idx] in MATH_DATA_SOURCES
        sample_mode = math_mode if is_math else requested_mode
        if not is_math and requested_mode == "gold_context" and (
            not str(questions[idx]).strip() or not str(gold_contexts[idx]).strip()
        ):
            sample_mode = fallback_mode
        privileged_text, messages, effective_mode = build_teacher_messages(
            problem=problem,
            answer=answer,
            solution=solution,
            mode=sample_mode,
            wrapper_variant=wrapper_variant,
            thinking=use_thinking,
            question=questions[idx],
            privileged_context=gold_contexts[idx],
            student_system_prompt=(
                student_system_prompts[idx]
                if student_system_prompts is not None
                else None),
            view_tail=resolve_view_prompt_tail(
                _custom_cfg_get(self, "view_prompt_tail_preset", "none")))
        privileged_texts.append(privileged_text)
        effective_modes.append(effective_mode)
        teacher_prompts.append(
            _render_chat(self, messages, **_teacher_chat_template_kwargs(self))
        )

    return {
        "privileged_texts": privileged_texts,
        "teacher_prompts": teacher_prompts,
        "effective_modes": effective_modes,
        "rationales": privileged_texts,
        "questions": questions,
        "gold_contexts": gold_contexts,
        "data_sources": data_sources}


def _build_evidence_removed_context_payloads(
    self,
    questions,
    minusgold_contexts,
    data_sources,
    raw_prompts,
):
    """Build current-policy prompts used to measure positive evidence support."""
    batch_size = len(questions)
    if not (
        len(minusgold_contexts) == len(data_sources) == batch_size
    ):
        raise AssertionError("Evidence-ablation fields are not batch-aligned")

    raw_prompts = (
        _collapse_repeated_to_batch(raw_prompts, batch_size)
        if raw_prompts is not None
        else [None] * batch_size
    )
    system_prompts = _student_system_prompts(raw_prompts, batch_size)
    if system_prompts is None:
        system_prompts = [None] * batch_size

    prompts = []
    eligible = []
    for idx, (question, minusgold_context, data_source) in enumerate(
        zip(questions, minusgold_contexts, data_sources)
    ):
        is_math = data_source in MATH_DATA_SOURCES
        if is_math:
            raw_prompt = raw_prompts[idx]
            if isinstance(raw_prompt, np.ndarray):
                raw_prompt = raw_prompt.tolist()
            messages = (
                raw_prompt
                if isinstance(raw_prompt, list)
                else build_rollout_messages(question, data_source=data_source)
            )
            eligible.append(False)
        else:
            if not str(question).strip() or not str(minusgold_context).strip():
                raise ValueError(
                    "eg_rlsd requires non-empty extra_info.question and "
                    "extra_info.minusgold_context for every non-math sample "
                    f"(batch index {idx}, data_source={data_source!r})."
                )
            messages = build_evidence_removed_messages(
                question=question,
                minusgold_context=minusgold_context,
                student_system_prompt=system_prompts[idx],
                envelope_header=_raw_prompt_envelope_header(raw_prompts[idx]),
                view_tail=resolve_view_prompt_tail(
                    _custom_cfg_get(self, "view_prompt_tail_preset", "none")),
            )
            eligible.append(True)

        prompts.append(
            _render_chat(self, messages, **_student_chat_template_kwargs(self))
        )

    return {
        "evidence_removed_prompts": prompts,
        "evidence_gate_eligible": eligible,
    }


def _build_teacher_noevidence_context_payloads(
    self,
    questions,
    data_sources,
    raw_prompts,
    require_question=False,
):
    """Build teacher prompts with the privileged evidence removed.

    Teacher-side counterfactual for gamma_T (eg_rlsd) and the CE methods. The
    rendered prompts go through a
    ``prefix="teacher_noevidence"`` batch and must be forwarded with the
    *teacher* weights, not the live policy, otherwise the contrast also
    absorbs teacher/policy weight drift.

    ``require_question=True`` (CE methods) raises on non-math rows with an
    empty question instead of silently rendering the plain rollout prompt:
    the question-only view is required by the method, and a silent fallback
    would degrade the whole run without any error in the logs.
    """
    batch_size = len(questions)
    if len(data_sources) != batch_size:
        raise AssertionError("Teacher no-evidence fields are not batch-aligned")

    raw_prompts = (
        _collapse_repeated_to_batch(raw_prompts, batch_size)
        if raw_prompts is not None
        else [None] * batch_size
    )
    system_prompts = _student_system_prompts(raw_prompts, batch_size)
    if system_prompts is None:
        system_prompts = [None] * batch_size

    prompts = []
    eligible = []
    for idx, (question, data_source) in enumerate(zip(questions, data_sources)):
        is_math = data_source in MATH_DATA_SOURCES
        if not is_math and not str(question).strip() and require_question:
            raise ValueError(
                "CE-based loss requires non-empty extra_info.question for every "
                f"non-math sample (batch index {idx}, data_source={data_source!r})."
            )
        if is_math or not str(question).strip():
            # Math rows are never gate-eligible; render the plain rollout
            # prompt so the tensor stays batch-aligned and is simply unused.
            raw_prompt = raw_prompts[idx]
            if isinstance(raw_prompt, np.ndarray):
                raw_prompt = raw_prompt.tolist()
            messages = (
                raw_prompt
                if isinstance(raw_prompt, list)
                else build_rollout_messages(question, data_source=data_source)
            )
            eligible.append(False)
        else:
            ptr_random_contexts = getattr(self, "_ptr_random_contexts", None)
            ptr_r = ""
            if (_resolve_teacher_view(self) in ("ptr_contrastive", "ptr_placebo")
                    and ptr_random_contexts is not None
                    and idx < len(ptr_random_contexts)):
                ptr_r = str(ptr_random_contexts[idx]).strip()
            if ptr_r:
                # CE contrast vs the SAME gold envelope but pointer->random:
                # delta = T(full+ptr->gold) - T(full+ptr->random) = c_ptr.
                _, messages, _ = build_teacher_messages(
                    problem="",
                    answer="",
                    solution="",
                    mode="gold_context",
                    question=question,
                    privileged_context=ptr_r,
                    student_system_prompt=system_prompts[idx],
                    wrapper_variant=_custom_cfg_get(self, "teacher_wrapper_variant", "neutral"),
                    view_tail=resolve_view_prompt_tail(
                        _custom_cfg_get(self, "view_prompt_tail_preset", "none")),
                )
            else:
                messages = build_teacher_noevidence_messages(
                    question=question,
                    student_system_prompt=system_prompts[idx],
                    view_tail=resolve_view_prompt_tail(
                        _custom_cfg_get(self, "view_prompt_tail_preset", "none")),
                )
            eligible.append(True)
        prompts.append(
            _render_chat(self, messages, **_student_chat_template_kwargs(self))
        )
    return {
        "teacher_noevidence_prompts": prompts,
        "teacher_noevidence_eligible": eligible,
    }


def _generate_rationales(self, batch: DataProto, gen_output: DataProto) -> dict | None:
    """Cache GT privileged text from the dataset batch for teacher forwarding."""
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    if loss_mode not in DISTILL_LOSS_MODES:
        return None
    if loss_mode in RLCSD_LOSS_MODES:
        return None

    raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    if raw_prompts is None:
        raw_prompts = gen_output.non_tensor_batch.get("raw_prompt")

    problems, ground_truths, solutions, questions, gold_contexts, data_sources = self._extract_problems_answers_solutions(
        batch,
        raw_prompts=raw_prompts,
        reward_models=batch.non_tensor_batch.get("reward_model"),
        extra_infos=batch.non_tensor_batch.get("extra_info"),
        include_long_context=True)
    payloads = _build_teacher_context_payloads(
        self,
        problems,
        ground_truths,
        solutions,
        questions=questions,
        gold_contexts=gold_contexts,
        data_sources=data_sources,
        student_system_prompts=_student_system_prompts(raw_prompts, len(problems)))
    payloads.update({
        "problems": problems,
        "ground_truths": ground_truths,
        "solutions": solutions,
        "questions": questions,
        "gold_contexts": gold_contexts,
        "data_sources": data_sources})
    return payloads


def _align_resp_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate the response dimension (dim=1) to match target_len."""
    current_len = tensor.shape[1]
    if current_len == target_len:
        return tensor
    if current_len < target_len:
        return F.pad(tensor, (0, target_len - current_len), value=0.0)
    return tensor[:, :target_len]


def _response_valid_lengths(batch: DataProto, responses: torch.Tensor) -> list[int]:
    """Per-row valid response lengths from the rollout mask.

    Must match ``response_mask.sum(dim=1)`` used by the actor's
    ``_align_compact_response_tensor``: filtering pad tokens by VALUE can
    drop a genuine pad-id token (e.g. Qwen's <|endoftext|>) that sits inside
    the mask and silently shift that row's teacher log-probs by one position.
    """
    response_length = responses.shape[1]
    mask = batch.batch.get("response_mask")
    if mask is None:
        attention_mask = batch.batch.get("attention_mask")
        if attention_mask is not None:
            mask = attention_mask[:, -response_length:]
    if mask is None:
        raise ValueError(
            "Cannot derive response lengths: batch has neither response_mask "
            "nor attention_mask."
        )
    return mask.sum(dim=1).to(dtype=torch.long).tolist()


def _build_teacher_batch_from_prompts(self, teacher_prompts, responses, prefix="teacher",
                                      response_lengths=None) -> DataProto:
    """Pack teacher prompts and sampled responses into model-ready tensors."""
    batch_size = len(teacher_prompts)
    if batch_size == 0:
        return DataProto.from_dict(tensors={})

    pad_id = self.tokenizer.pad_token_id or 0
    teacher_prompt_ids_list = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in teacher_prompts]

    teacher_input_ids_list = []
    teacher_prompt_lens = []
    teacher_total_budget = _teacher_total_token_budget(self)
    for i in range(batch_size):
        prompt_ids = teacher_prompt_ids_list[i]
        if response_lengths is not None:
            # Slice by mask length so the teacher's response token count is
            # identical to response_mask.sum() by construction.
            response_ids = responses[i][: int(response_lengths[i])].tolist()
        else:
            # Legacy fallback (no mask available): filter pads by value.
            response_ids = [token for token in responses[i].tolist() if token != pad_id]
        max_prompt_tokens = max(teacher_total_budget - len(response_ids), 0)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:] if max_prompt_tokens > 0 else []
        teacher_input_ids_list.append(prompt_ids + response_ids)
        teacher_prompt_lens.append(len(prompt_ids))

    max_prompt_len = max(teacher_prompt_lens)
    response_len = max(len(ids) - teacher_prompt_lens[i] for i, ids in enumerate(teacher_input_ids_list))
    total_len = max_prompt_len + response_len

    teacher_input_ids = torch.full((batch_size, total_len), pad_id, dtype=torch.long)
    teacher_attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long)
    teacher_position_ids = torch.zeros((batch_size, total_len), dtype=torch.long)

    for i, ids in enumerate(teacher_input_ids_list):
        prompt_len = teacher_prompt_lens[i]
        prompt_offset = max_prompt_len - prompt_len
        response_ids = ids[prompt_len:]
        response_count = len(response_ids)

        teacher_input_ids[i, prompt_offset:max_prompt_len] = torch.tensor(ids[:prompt_len], dtype=torch.long)
        if response_count > 0:
            teacher_input_ids[i, max_prompt_len:max_prompt_len + response_count] = torch.tensor(
                response_ids, dtype=torch.long
            )

        teacher_attention_mask[i, prompt_offset:max_prompt_len + response_count] = 1
        teacher_position_ids[i, prompt_offset:max_prompt_len + response_count] = torch.arange(
            prompt_len + response_count
        )

    return DataProto.from_dict(
        tensors={
            f"{prefix}_input_ids": teacher_input_ids,
            f"{prefix}_attention_mask": teacher_attention_mask,
            f"{prefix}_position_ids": teacher_position_ids}
    )


def _build_teacher_multi_batch_from_prompt_groups(self, teacher_prompt_groups, responses, prefix="teacher",
                                                  response_lengths=None) -> DataProto:
    """Pack per-sample teacher prompt groups into [batch, num_ctx, seq] tensors."""
    batch_size = len(teacher_prompt_groups)
    if batch_size == 0:
        return DataProto.from_dict(tensors={})

    group_sizes = [len(group) for group in teacher_prompt_groups]
    max_group_size = max(group_sizes, default=0)
    if max_group_size <= 0:
        raise ValueError(f"{prefix} received no teacher prompts to pack.")

    # Replicate the first prompt of each sample to fill empty slots so every
    # (sample, ctx) carries a non-empty input. This lets the actor's
    # multi-context teacher forward run unconditionally on every column,
    # avoiding NCCL collective desync when DP ranks see different valid_mask
    # column populations. Replicas are marked invalid in valid_mask so the
    # downstream loss ignores their outputs.
    flat_prompts = []
    flat_responses = []
    flat_response_lengths = [] if response_lengths is not None else None
    flat_indices = []
    for sample_idx, prompt_group in enumerate(teacher_prompt_groups):
        if not prompt_group:
            raise ValueError(f"{prefix} sample {sample_idx} has an empty prompt group.")
        for group_idx in range(max_group_size):
            real = group_idx < len(prompt_group)
            prompt = prompt_group[group_idx] if real else prompt_group[0]
            flat_prompts.append(prompt)
            flat_responses.append(responses[sample_idx])
            if flat_response_lengths is not None:
                flat_response_lengths.append(response_lengths[sample_idx])
            flat_indices.append((sample_idx, group_idx, real))

    flat_responses = torch.stack(flat_responses, dim=0)
    flat_prefix = f"{prefix}_flat"
    flat_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=flat_prompts,
        responses=flat_responses,
        prefix=flat_prefix,
        response_lengths=flat_response_lengths)

    flat_input_ids = flat_batch.batch[f"{flat_prefix}_input_ids"]
    flat_attention_mask = flat_batch.batch[f"{flat_prefix}_attention_mask"]
    flat_position_ids = flat_batch.batch[f"{flat_prefix}_position_ids"]
    seq_len = flat_input_ids.shape[1]

    input_ids = flat_input_ids.new_zeros((batch_size, max_group_size, seq_len))
    attention_mask = flat_attention_mask.new_zeros((batch_size, max_group_size, seq_len))
    position_ids = flat_position_ids.new_zeros((batch_size, max_group_size, seq_len))
    valid_mask = torch.zeros((batch_size, max_group_size), dtype=torch.bool)

    for flat_idx, (sample_idx, group_idx, real) in enumerate(flat_indices):
        input_ids[sample_idx, group_idx] = flat_input_ids[flat_idx]
        attention_mask[sample_idx, group_idx] = flat_attention_mask[flat_idx]
        position_ids[sample_idx, group_idx] = flat_position_ids[flat_idx]
        if real:
            valid_mask[sample_idx, group_idx] = True

    return DataProto.from_dict(
        tensors={
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attention_mask": attention_mask,
            f"{prefix}_position_ids": position_ids,
            f"{prefix}_valid_mask": valid_mask},
        num_batch_dims=1)


def _compute_rlcsd_teacher_inputs(self, batch: DataProto) -> DataProto | None:
    """Build positive/negative teacher prompts from rollout groups identified by uid."""
    self._data_metric_overrides = _compute_prefilter_metric_overrides(self, batch)
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    if loss_mode not in RLCSD_LOSS_MODES:
        raise ValueError(f"_compute_rlcsd_teacher_inputs called with unsupported loss_mode={loss_mode}")
    batch_size = batch.batch.batch_size[0]
    if batch_size == 0:
        batch.meta_info["skip_actor_update"] = True
        self._aligned_teacher_rollout_details = {}
        self._rlcsd_step_metrics = {"rlcsd/skip_actor_update": 1.0}
        return None

    uid_values = batch.non_tensor_batch.get("uid")
    if uid_values is None:
        raise ValueError("rlcsd_* requires uid in non_tensor_batch to recover rollout groups.")

    problems, ground_truths, _ = self._extract_problems_answers_solutions(batch)
    responses = batch.batch["responses"]
    responses_text = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    reward_tensor = batch.batch["token_level_scores"] if "token_level_scores" in batch.batch.keys() else batch.batch["rm_scores"]
    if reward_tensor.dim() > 1:
        rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
    else:
        rewards = reward_tensor.detach().cpu().tolist()

    groups = {}
    for idx, uid in enumerate(uid_values.tolist()):
        groups.setdefault(str(uid), []).append(idx)

    num_aborted = 0
    all_same_outcome_group_count = 0
    self_negative_missing_boxed_count = 0
    self_excluded_sampling_skip_count = 0
    valid_mask = np.zeros(batch_size, dtype=bool)
    correct_prompt_by_idx = {}
    wrong_prompt_by_idx = {}
    correct_text_by_idx = {}
    wrong_text_by_idx = {}
    correct_answer_by_idx = {}
    wrong_answer_by_idx = {}
    correct_prompt_group_by_idx = {}
    wrong_prompt_group_by_idx = {}
    correct_text_group_by_idx = {}
    wrong_text_group_by_idx = {}
    correct_answer_group_by_idx = {}
    wrong_answer_group_by_idx = {}
    correct_pool_sizes = []
    wrong_pool_sizes = []
    valid_group_count = 0
    for group_indices in groups.values():
        positives = [idx for idx in group_indices if rewards[idx] > 0]
        negatives = [idx for idx in group_indices if rewards[idx] <= 0]
        if not positives or not negatives:
            num_aborted += 1
            all_same_outcome_group_count += 1
            continue

        problem = problems[group_indices[0]]
        positive_contexts = {}
        for idx in positives:
            correct_answer = extract_boxed_answer(responses_text[idx]) or ground_truths[idx]
            positive_contexts[idx] = _materialize_teacher_context(
                self,
                problem=problem,
                answer=correct_answer,
                solution=responses_text[idx])

        negative_contexts = {}
        for idx in negatives:
            wrong_answer = extract_boxed_answer(responses_text[idx])
            if wrong_answer is None:
                continue
            negative_contexts[idx] = _materialize_teacher_context(
                self,
                problem=problem,
                answer=wrong_answer,
                solution=responses_text[idx])

        valid_positive_indices = list(positive_contexts)
        valid_negative_indices = list(negative_contexts)
        if not valid_positive_indices or not valid_negative_indices:
            num_aborted += 1
            continue

        group_valid_count = 0
        for idx in group_indices:
            if idx in positive_contexts:
                positive_idx = _sample_other_rollout(valid_positive_indices, idx)
                if positive_idx is None:
                    self_excluded_sampling_skip_count += 1
                    continue
                correct_context = positive_contexts[positive_idx]
                wrong_idx = random.choice(valid_negative_indices)
                wrong_context = negative_contexts[wrong_idx]
                rlcsd_wrong_pool = list(valid_negative_indices)
            else:
                negative_idx = _sample_other_rollout(valid_negative_indices, idx)
                if negative_idx is None:
                    self_excluded_sampling_skip_count += 1
                    continue
                wrong_context = negative_contexts.get(negative_idx)
                if wrong_context is None:
                    self_negative_missing_boxed_count += 1
                    continue
                wrong_idx = negative_idx
                correct_context = positive_contexts[random.choice(valid_positive_indices)]
                rlcsd_wrong_pool = [c for c in valid_negative_indices if c != idx]

            valid_mask[idx] = True
            correct_prompt_by_idx[idx] = correct_context["prompt"]
            wrong_prompt_by_idx[idx] = wrong_context["prompt"]
            correct_text_by_idx[idx] = correct_context["text"]
            wrong_text_by_idx[idx] = wrong_context["text"]
            correct_answer_by_idx[idx] = correct_context["answer"]
            wrong_answer_by_idx[idx] = wrong_context["answer"]
            if loss_mode == "rlcsd":
                # K = min(rlcsd_k_max, |non-self negatives|). Always include the
                # chosen single-pair wrong sibling first so the marginal mean
                # covers it; fill remaining slots with random other non-self
                # negatives.
                k_max = int(_cfg_get(self, "rlcsd_k_max", RLCSD_DEFAULT_K_MAX) or RLCSD_DEFAULT_K_MAX)
                rest = [c for c in rlcsd_wrong_pool if c != wrong_idx]
                random.shuffle(rest)
                chosen = [wrong_idx] + rest[: max(k_max - 1, 0)]
                wrong_prompt_group_by_idx[idx] = [negative_contexts[c]["prompt"] for c in chosen]
                wrong_text_group_by_idx[idx]   = [negative_contexts[c]["text"]   for c in chosen]
                wrong_answer_group_by_idx[idx] = [negative_contexts[c]["answer"] for c in chosen]
                wrong_pool_sizes.append(len(chosen))
            group_valid_count += 1

        if group_valid_count > 0:
            valid_group_count += 1
        else:
            num_aborted += 1

    num_groups = len(groups)
    valid_sample_count = int(valid_mask.sum())
    step_metrics = {
        "rlcsd/aborted_ratio": float(num_aborted / max(num_groups, 1)),
        "rlcsd/valid_group_ratio": float(valid_group_count / max(num_groups, 1)),
        "rlcsd/valid_group_count": float(valid_group_count),
        "rlcsd/valid_sample_count": float(valid_sample_count),
        "rlcsd/group_count": float(num_groups),
        "rlcsd/all_same_outcome_group_count": float(all_same_outcome_group_count),
        "rlcsd/self_negative_missing_boxed_count": float(self_negative_missing_boxed_count),
        "rlcsd/self_excluded_sampling_skip_count": float(self_excluded_sampling_skip_count),
        "rlcsd/skipped_sample_count": float(batch_size - valid_sample_count),
        "rlcsd/actor_pad_count": 0.0,
        "rlcsd/actor_padded_batch_size": float(valid_sample_count),
        "rlcsd/correct_pool_size_mean": float(np.mean(correct_pool_sizes)) if correct_pool_sizes else 0.0,
        "rlcsd/wrong_pool_size_mean": float(np.mean(wrong_pool_sizes)) if wrong_pool_sizes else 0.0}

    if not valid_mask.any():
        filtered_batch = batch.select_idxs(valid_mask)
        batch.batch = filtered_batch.batch
        batch.non_tensor_batch = filtered_batch.non_tensor_batch
        batch.meta_info["skip_actor_update"] = True
        step_metrics["rlcsd/skip_actor_update"] = 1.0
        self._aligned_teacher_rollout_details = {
            "teacher_correct_prompts": [],
            "teacher_wrong_prompts": [],
            "privileged_texts_correct": [],
            "privileged_texts_wrong": [],
            "correct_answers": [],
            "wrong_answers": []}
        self._rlcsd_step_metrics = step_metrics
        return None

    kept_indices = [idx for idx in range(batch_size) if valid_mask[idx]]
    filtered_batch = batch.select_idxs(valid_mask)
    batch.batch = filtered_batch.batch
    batch.non_tensor_batch = filtered_batch.non_tensor_batch

    if loss_mode == "rlcsd":
        teacher_correct_prompts = [correct_prompt_by_idx[idx] for idx in kept_indices]
        teacher_wrong_prompts = [wrong_prompt_by_idx[idx] for idx in kept_indices]
        privileged_texts_correct = [correct_text_by_idx[idx] for idx in kept_indices]
        privileged_texts_wrong = [wrong_text_by_idx[idx] for idx in kept_indices]
        correct_answers = [correct_answer_by_idx[idx] for idx in kept_indices]
        wrong_answers = [wrong_answer_by_idx[idx] for idx in kept_indices]
        # rlcsd_5 only marginalizes on the wrong side; correct stays single.
        teacher_correct_prompt_groups = []
        privileged_text_groups_correct = []
        correct_answer_groups = []
        teacher_wrong_prompt_groups = [wrong_prompt_group_by_idx[idx] for idx in kept_indices]
        privileged_text_groups_wrong = [wrong_text_group_by_idx[idx] for idx in kept_indices]
        wrong_answer_groups = [wrong_answer_group_by_idx[idx] for idx in kept_indices]
    else:
        teacher_correct_prompts = [correct_prompt_by_idx[idx] for idx in kept_indices]
        teacher_wrong_prompts = [wrong_prompt_by_idx[idx] for idx in kept_indices]
        privileged_texts_correct = [correct_text_by_idx[idx] for idx in kept_indices]
        privileged_texts_wrong = [wrong_text_by_idx[idx] for idx in kept_indices]
        correct_answers = [correct_answer_by_idx[idx] for idx in kept_indices]
        wrong_answers = [wrong_answer_by_idx[idx] for idx in kept_indices]
        teacher_correct_prompt_groups = []
        teacher_wrong_prompt_groups = []
        privileged_text_groups_correct = []
        privileged_text_groups_wrong = []
        correct_answer_groups = []
        wrong_answer_groups = []

    self._aligned_teacher_rollout_details = {
        "teacher_correct_prompts": teacher_correct_prompts,
        "teacher_wrong_prompts": teacher_wrong_prompts,
        "privileged_texts_correct": privileged_texts_correct,
        "privileged_texts_wrong": privileged_texts_wrong,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "teacher_correct_prompt_groups": teacher_correct_prompt_groups,
        "teacher_wrong_prompt_groups": teacher_wrong_prompt_groups,
        "privileged_text_groups_correct": privileged_text_groups_correct,
        "privileged_text_groups_wrong": privileged_text_groups_wrong,
        "correct_answer_groups": correct_answer_groups,
        "wrong_answer_groups": wrong_answer_groups,
        "effective_modes": ["solution_answer"] * len(teacher_correct_prompts)}
    self._rlcsd_step_metrics = step_metrics

    rlcsd_response_lengths = _response_valid_lengths(batch, batch.batch["responses"])
    if loss_mode == "rlcsd":
        teacher_correct_pair = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_correct_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_correct",
            response_lengths=rlcsd_response_lengths)
        teacher_wrong_multi = _build_teacher_multi_batch_from_prompt_groups(
            self,
            teacher_prompt_groups=teacher_wrong_prompt_groups,
            responses=batch.batch["responses"],
            prefix="teacher_wrong_multi",
            response_lengths=rlcsd_response_lengths)
        return teacher_correct_pair.union(teacher_wrong_multi)
    else:
        teacher_correct = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_correct_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_correct",
            response_lengths=rlcsd_response_lengths)
        teacher_wrong = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_wrong_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_wrong",
            response_lengths=rlcsd_response_lengths)
        return teacher_correct.union(teacher_wrong)


# ------------------------------------------------------------------ #
# Step 3: Compute teacher log probs (called during training phase)
# ------------------------------------------------------------------ #

def _compute_teacher_log_probs(self, batch: DataProto) -> DataProto | None:
    """Build teacher-side inputs from cached GT privileged prompts."""
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    self._data_metric_overrides = {}
    if loss_mode not in DISTILL_LOSS_MODES:
        return None
    if loss_mode in RLCSD_LOSS_MODES:
        return _compute_rlcsd_teacher_inputs(self, batch)

    self._rlcsd_step_metrics = {}

    batch_size = batch.batch.batch_size[0]
    raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    extracted = self._extract_problems_answers_solutions(
        batch,
        raw_prompts=raw_prompts,
        include_long_context=True,
        include_evidence_ablation=loss_mode in EVIDENCE_GATED_LOSS_MODES,
    )
    if loss_mode in EVIDENCE_GATED_LOSS_MODES:
        (
            problems,
            ground_truths,
            solutions,
            questions,
            gold_contexts,
            minusgold_contexts,
            data_sources,
        ) = extracted
    else:
        problems, ground_truths, solutions, questions, gold_contexts, data_sources = extracted
        minusgold_contexts = [""] * batch_size
    aligned_context = _build_teacher_context_payloads(
        self,
        problems,
        ground_truths,
        solutions,
        questions=questions,
        gold_contexts=gold_contexts,
        data_sources=data_sources,
        student_system_prompts=_student_system_prompts(raw_prompts, len(problems)))
    self._aligned_teacher_rollout_details = aligned_context
    teacher_prompts = aligned_context["teacher_prompts"]

    if len(teacher_prompts) != batch_size:
        rollout_n = self.config.actor_rollout_ref.rollout.n
        teacher_prompts = _expand_to_rollout(teacher_prompts, batch_size, rollout_n)

    responses = batch.batch["responses"]
    response_valid_lengths = _response_valid_lengths(batch, responses)

    teacher_prompt_ids_list = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in teacher_prompts]
    teacher_prompt_lens = []
    teacher_prompt_lens_raw = []
    teacher_prompt_tokens_truncated = []
    teacher_prompt_truncated = []
    teacher_total_budget = _teacher_total_token_budget(self)
    for i in range(batch_size):
        raw_prompt_len = len(teacher_prompt_ids_list[i])
        max_prompt_tokens = max(teacher_total_budget - int(response_valid_lengths[i]), 0)
        kept_prompt_len = min(raw_prompt_len, max_prompt_tokens)
        truncated_tokens = raw_prompt_len - kept_prompt_len
        teacher_prompt_lens.append(kept_prompt_len)
        teacher_prompt_lens_raw.append(raw_prompt_len)
        teacher_prompt_tokens_truncated.append(truncated_tokens)
        teacher_prompt_truncated.append(1.0 if truncated_tokens > 0 else 0.0)

    batch.non_tensor_batch["teacher_prompt_truncated"] = np.asarray(teacher_prompt_truncated, dtype=np.float32)
    batch.non_tensor_batch["teacher_prompt_truncated_tokens"] = np.asarray(
        teacher_prompt_tokens_truncated, dtype=np.float32
    )
    batch.non_tensor_batch["teacher_prompt_tokens_before_trunc"] = np.asarray(
        teacher_prompt_lens_raw, dtype=np.float32
    )
    batch.non_tensor_batch["teacher_prompt_tokens_after_trunc"] = np.asarray(
        teacher_prompt_lens, dtype=np.float32
    )
    teacher_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=teacher_prompts,
        responses=responses,
        prefix="teacher",
        response_lengths=response_valid_lengths)
    # distill_coef_correct >= 0 (asymmetric per-side mu, 2026-08-31) needs the
    # same reward-derived wrong-trajectory mask as distill_wrong_only.
    _needs_wrong_mask = _cfg_bool(self, "distill_wrong_only", False) or (
        float(_cfg_get(self, "distill_coef_correct", -1.0)) >= 0.0
    )
    if loss_mode in ("opsd", "grpo_opsd") and _needs_wrong_mask:
        score_tensor = batch.batch.get("token_level_scores")
        if score_tensor is None:
            raise ValueError(
                "distill_wrong_only=True requires token_level_scores after reward computation."
            )
        sequence_scores = (
            score_tensor.sum(dim=-1) if score_tensor.dim() > 1 else score_tensor
        ).float()
        if not torch.isfinite(sequence_scores).all():
            raise ValueError("distill_wrong_only received non-finite sequence scores.")
        binary_scores = torch.isclose(sequence_scores, torch.zeros_like(sequence_scores)) | torch.isclose(
            sequence_scores, torch.ones_like(sequence_scores)
        )
        if not binary_scores.all():
            unique_scores = torch.unique(sequence_scores).detach().cpu().tolist()
            raise ValueError(
                "distill_wrong_only expects binary final correctness rewards; "
                f"got sequence scores {unique_scores[:16]}"
            )
        teacher_batch.batch["distill_sample_mask"] = sequence_scores < 0.5
    use_cf_gate = _cfg_bool(self, "distill_key_ablation_gate", False)
    certified_only = use_cf_gate or _cfg_bool(self, "distill_certified_only", False)
    if certified_only:
        if loss_mode not in ("opsd", "grpo_opsd") or _resolve_teacher_view(self) != "gold_only":
            raise ValueError("Certified distillation requires OPSD with the gold-only teacher")
        keyabl_contexts, keyabl_eligible = _extract_key_ablation_contexts(
            self, batch, gold_contexts, data_sources
        )
        if len(keyabl_eligible) != batch_size:
            keyabl_eligible = _expand_to_rollout(
                keyabl_eligible, batch_size, self.config.actor_rollout_ref.rollout.n
            )
        teacher_batch.batch["distill_key_ablation_sample_mask"] = torch.tensor(keyabl_eligible, dtype=torch.bool)
        aligned_context["keyabl_gate_eligible"] = keyabl_eligible

    if use_cf_gate:
        keyabl_payloads = _build_teacher_context_payloads(
            self,
            problems,
            ground_truths,
            solutions,
            questions=questions,
            gold_contexts=keyabl_contexts,
            data_sources=data_sources,
            student_system_prompts=_student_system_prompts(
                raw_prompts, len(problems)
            ),
        )
        teacher_keyabl_prompts = keyabl_payloads["teacher_prompts"]
        if len(teacher_keyabl_prompts) != batch_size:
            rollout_n = self.config.actor_rollout_ref.rollout.n
            teacher_keyabl_prompts = _expand_to_rollout(
                teacher_keyabl_prompts, batch_size, rollout_n
            )
        aligned_context["teacher_keyabl_prompts"] = teacher_keyabl_prompts
        keyabl_batch = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_keyabl_prompts,
            responses=responses,
            prefix="teacher_keyabl",
            response_lengths=response_valid_lengths,
        )
        teacher_batch = teacher_batch.union(keyabl_batch)
    if loss_mode in TWO_TEACHER_VIEW_LOSS_MODES:
        # CE methods contrast teacher(gold+q) against teacher(q only). Only
        # the question-only teacher view is added; there is no
        # evidence_removed (minusgold) policy forward at all.
        noevidence_context = _build_teacher_noevidence_context_payloads(
            self,
            questions=questions,
            data_sources=data_sources,
            raw_prompts=raw_prompts,
            require_question=True,
        )
        teacher_noevidence_prompts = noevidence_context["teacher_noevidence_prompts"]
        noevidence_eligible = noevidence_context["teacher_noevidence_eligible"]
        if len(teacher_noevidence_prompts) != batch_size:
            rollout_n = self.config.actor_rollout_ref.rollout.n
            teacher_noevidence_prompts = _expand_to_rollout(
                teacher_noevidence_prompts, batch_size, rollout_n
            )
            noevidence_eligible = _expand_to_rollout(
                noevidence_eligible, batch_size, rollout_n
            )
        aligned_context["evidence_gate_eligible"] = noevidence_eligible
        noevidence_batch = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_noevidence_prompts,
            responses=responses,
            prefix="teacher_noevidence",
            response_lengths=response_valid_lengths,
        )
        noevidence_batch.batch["evidence_sample_mask"] = torch.tensor(
            noevidence_eligible, dtype=torch.bool
        )
        return teacher_batch.union(noevidence_batch)
    if loss_mode not in EVIDENCE_GATED_LOSS_MODES:
        if loss_mode in ("rlsd", "opsd", "grpo_opsd") and _cfg_bool(
                self, "rlsd_skip_math", False):
            # Per-row eligibility mask (True = non-math), mirroring the EG/CE
            # eligibility contract. rlsd: math rows fall back to exact GRPO in
            # compute_policy_loss_rlsd (w=1). opsd/grpo_opsd: math rows are
            # dropped from the distill average in _distill_eligibility_mask,
            # which for grpo_opsd also means exact GRPO on those rows since the
            # PG half never reads the mask. data_sources comes from
            # _extract_problems_answers_solutions and is already batch-sized.
            if certified_only:
                rlsd_eligible = keyabl_eligible
            else:
                rlsd_eligible = [
                    source not in MATH_DATA_SOURCES for source in data_sources
                ]
            teacher_batch.batch["evidence_sample_mask"] = torch.tensor(
                rlsd_eligible, dtype=torch.bool
            )
        return teacher_batch

    evidence_context = _build_evidence_removed_context_payloads(
        self,
        questions=questions,
        minusgold_contexts=minusgold_contexts,
        data_sources=data_sources,
        raw_prompts=raw_prompts,
    )
    evidence_removed_prompts = evidence_context["evidence_removed_prompts"]
    evidence_gate_eligible = evidence_context["evidence_gate_eligible"]
    if len(evidence_removed_prompts) != batch_size:
        rollout_n = self.config.actor_rollout_ref.rollout.n
        evidence_removed_prompts = _expand_to_rollout(
            evidence_removed_prompts, batch_size, rollout_n
        )
        evidence_gate_eligible = _expand_to_rollout(
            evidence_gate_eligible, batch_size, rollout_n
        )

    aligned_context["evidence_gate_eligible"] = evidence_gate_eligible
    evidence_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=evidence_removed_prompts,
        responses=responses,
        prefix="evidence_removed",
        response_lengths=response_valid_lengths,
    )
    evidence_batch.batch["evidence_sample_mask"] = torch.tensor(
        evidence_gate_eligible, dtype=torch.bool
    )
    combined = teacher_batch.union(evidence_batch)

    teacher_gate_mode = _normalize_teacher_gate_mode(
        _custom_cfg_get(self, "teacher_gate_mode", "off")
    )
    if teacher_gate_mode == "off":
        return combined

    noevidence_context = _build_teacher_noevidence_context_payloads(
        self,
        questions=questions,
        data_sources=data_sources,
        raw_prompts=raw_prompts,
    )
    teacher_noevidence_prompts = noevidence_context["teacher_noevidence_prompts"]
    if len(teacher_noevidence_prompts) != batch_size:
        rollout_n = self.config.actor_rollout_ref.rollout.n
        teacher_noevidence_prompts = _expand_to_rollout(
            teacher_noevidence_prompts, batch_size, rollout_n
        )
    noevidence_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=teacher_noevidence_prompts,
        responses=responses,
        prefix="teacher_noevidence",
        response_lengths=response_valid_lengths,
    )
    return combined.union(noevidence_batch)


def _judge_rationales(self) -> dict:
    """Return per-step teacher-context metrics when available."""
    metrics = getattr(self, "_rlcsd_step_metrics", {})
    self._rlcsd_step_metrics = {}
    return dict(metrics)


def _get_data_metric_overrides(self) -> dict:
    """Return per-step metric overrides that should win over post-filter stats."""
    metrics = getattr(self, "_data_metric_overrides", {})
    self._data_metric_overrides = {}
    return dict(metrics)


_ORIGINAL_RLHF_BUILD_MESSAGES = None


def _patched_rlhf_build_messages(self, example: dict):
    """Format stored raw problems into official OPSD train/val rollout prompts."""
    messages = _ORIGINAL_RLHF_BUILD_MESSAGES(self, example)

    data_source = str(example.get("data_source", "")).lower()
    if data_source not in MATH_DATA_SOURCES:
        return messages

    # contract-bearing datasets pass through: v5+ parquets embed the full
    # contract (system role) in the prompt messages. Rewrapping them dropped
    # the system prompt (chat template then injects the Qwen default) and
    # leaked the contract text into the user turn behind a second "Problem:"
    # prefix (bug found 2026-08-03 in the v5warmup rollouts). Only legacy
    # bare-problem math rows need the official OPSD wrapper below.
    if isinstance(messages, list) and any(
        isinstance(m, dict) and m.get("role") == "system" for m in messages
    ):
        return messages

    extra_info = example.get("extra_info") or {}
    problem = _content_to_text(extra_info.get("problem", "")) if isinstance(extra_info, dict) else ""
    if not problem:
        problem = _raw_prompt_to_problem(messages)

    # Read thinking_system_prompt from data config (self = RLHFDataset).
    use_thinking = False
    cfg = getattr(self, "config", None)
    if cfg is not None:
        val = cfg.get("thinking_system_prompt", False)
        use_thinking = str(val).strip().lower() in ("1", "true", "yes", "on") if isinstance(val, str) else bool(val)
    return build_rollout_messages(problem, data_source=data_source, thinking=use_thinking)


def _save_train_rollout(self, batch: DataProto):
    """Save train rollout details with teacher context aligned to the current batch order."""
    log_dir = self._get_rollout_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"train_step_{self.global_steps}.jsonl")

    batch_size = batch.batch.batch_size[0]
    prompts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    score_tensor = batch.batch["token_level_scores"]
    if score_tensor.dim() > 1:
        scores = score_tensor.sum(dim=-1).cpu().tolist()
    else:
        scores = score_tensor.cpu().tolist()

    ground_truths = []
    for i in range(batch_size):
        item = batch[i]
        reward_model = item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        ground_truths.append(str(ground_truth))

    aligned_context = getattr(self, "_aligned_teacher_rollout_details", None)
    privileged_texts = aligned_context.get("privileged_texts", []) if aligned_context is not None else []
    teacher_prompts = aligned_context.get("teacher_prompts", []) if aligned_context is not None else []
    effective_modes = aligned_context.get("effective_modes", []) if aligned_context is not None else []
    privileged_texts_correct = aligned_context.get("privileged_texts_correct", []) if aligned_context is not None else []
    privileged_texts_wrong = aligned_context.get("privileged_texts_wrong", []) if aligned_context is not None else []
    teacher_correct_prompts = aligned_context.get("teacher_correct_prompts", []) if aligned_context is not None else []
    teacher_wrong_prompts = aligned_context.get("teacher_wrong_prompts", []) if aligned_context is not None else []
    teacher_correct_prompt_groups = aligned_context.get("teacher_correct_prompt_groups", []) if aligned_context is not None else []
    teacher_wrong_prompt_groups = aligned_context.get("teacher_wrong_prompt_groups", []) if aligned_context is not None else []
    privileged_text_groups_correct = aligned_context.get("privileged_text_groups_correct", []) if aligned_context is not None else []
    privileged_text_groups_wrong = aligned_context.get("privileged_text_groups_wrong", []) if aligned_context is not None else []
    correct_answers = aligned_context.get("correct_answers", []) if aligned_context is not None else []
    wrong_answers = aligned_context.get("wrong_answers", []) if aligned_context is not None else []
    correct_answer_groups = aligned_context.get("correct_answer_groups", []) if aligned_context is not None else []
    wrong_answer_groups = aligned_context.get("wrong_answer_groups", []) if aligned_context is not None else []

    lines = []
    for i in range(batch_size):
        entry = {
            "step": self.global_steps,
            "sample_idx": i,
            "prompt": prompts[i],
            "response": responses[i],
            "score": scores[i],
            "ground_truth": ground_truths[i]}
        if i < len(privileged_texts):
            entry["privileged_text"] = privileged_texts[i]
        if i < len(teacher_prompts):
            entry["teacher_prompt"] = teacher_prompts[i]
        if i < len(effective_modes):
            entry["privileged_text_mode"] = effective_modes[i]
        if i < len(privileged_texts_correct):
            entry["privileged_text_correct"] = privileged_texts_correct[i]
        if i < len(privileged_texts_wrong):
            entry["privileged_text_wrong"] = privileged_texts_wrong[i]
        if i < len(teacher_correct_prompts):
            entry["teacher_correct_prompt"] = teacher_correct_prompts[i]
        if i < len(teacher_wrong_prompts):
            entry["teacher_wrong_prompt"] = teacher_wrong_prompts[i]
        if i < len(teacher_correct_prompt_groups):
            entry["teacher_correct_prompt_pool"] = teacher_correct_prompt_groups[i]
        if i < len(teacher_wrong_prompt_groups):
            entry["teacher_wrong_prompt_pool"] = teacher_wrong_prompt_groups[i]
        if i < len(privileged_text_groups_correct):
            entry["privileged_text_correct_pool"] = privileged_text_groups_correct[i]
        if i < len(privileged_text_groups_wrong):
            entry["privileged_text_wrong_pool"] = privileged_text_groups_wrong[i]
        if i < len(correct_answers):
            entry["teacher_correct_answer"] = correct_answers[i]
        if i < len(wrong_answers):
            entry["teacher_wrong_answer"] = wrong_answers[i]
        if i < len(correct_answer_groups):
            entry["teacher_correct_answer_pool"] = correct_answer_groups[i]
        if i < len(wrong_answer_groups):
            entry["teacher_wrong_answer_pool"] = wrong_answer_groups[i]
        lines.append(json.dumps(entry, ensure_ascii=False))

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved train rollout details to {filename} ({batch_size} samples)")


# ------------------------------------------------------------------ #
# Apply monkey patches
# ------------------------------------------------------------------ #

def _apply_patches():
    global _ORIGINAL_RLHF_BUILD_MESSAGES
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if _ORIGINAL_RLHF_BUILD_MESSAGES is None:
        _ORIGINAL_RLHF_BUILD_MESSAGES = RLHFDataset._build_messages
    RLHFDataset._build_messages = _patched_rlhf_build_messages

    RayPPOTrainer._extract_problems_answers_solutions = _extract_problems_answers_solutions
    RayPPOTrainer._generate_rationales = _generate_rationales
    RayPPOTrainer._compute_teacher_log_probs = _compute_teacher_log_probs
    RayPPOTrainer._prepare_actor_update_batch = _prepare_actor_update_batch
    RayPPOTrainer._judge_rationales = _judge_rationales
    RayPPOTrainer._get_data_metric_overrides = _get_data_metric_overrides
    RayPPOTrainer._save_train_rollout = _save_train_rollout


_apply_patches()


def _ensure_legacy_worker_impl(config):
    worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
    if worker_impl == "disable":
        with open_dict(config):
            config.trainer.use_legacy_worker_impl = "enable"
        print(
            "[Self-Distill] Overriding trainer.use_legacy_worker_impl=disable -> enable. "
            "Privileged teacher prompting and full-logit distillation are currently implemented "
            "only in the legacy actor worker path."
        )
    return config


# ------------------------------------------------------------------ #
# Custom TaskRunner that applies patches inside the Ray actor
# ------------------------------------------------------------------ #

from verl.trainer.main_ppo import TaskRunner


class SelfDistillTaskRunner(TaskRunner):
    def run(self, config):
        from src.self_distill_main import _apply_patches as apply_sd_patches

        apply_sd_patches()
        return super().run(config)


if __name__ == "__main__":
    import hydra
    import ray
    from verl.trainer.main_ppo import migrate_legacy_reward_impl, run_ppo
    from verl.utils.device import auto_set_device

    @hydra.main(config_path="../third_party/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
    def main(config):
        auto_set_device(config)
        config = migrate_legacy_reward_impl(config)
        config = _ensure_legacy_worker_impl(config)
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SelfDistillTaskRunner))

    main()
