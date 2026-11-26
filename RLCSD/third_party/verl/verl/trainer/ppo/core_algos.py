# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

import math
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    GDPO = "gdpo"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GDPO)  # or simply: @register_adv_est("gdpo")
def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    batch: Optional[dict] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GDPO: Group reward-Decoupled Normalization Policy Optimization.

    Instead of summing all reward dimensions first (like GRPO), GDPO normalizes
    each reward dimension independently within each group before aggregation.
    This prevents a dominant reward signal from drowning out weaker ones.

    Mathematical formulation:
        Step 1 – Group-wise decoupled normalization (via GRPO per dimension):
            For each reward dimension k, within each group g:
            A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)

        Step 2 – Weighted aggregation:
            A_sum = Σ_k w_k · A_k

        Step 3 – Batch-level normalization (via masked_whiten):
            A_final = whiten(A_sum, response_mask)

    Args:
        token_level_rewards: (bs, response_length) – standard token-level rewards.
            Used as fallback when per-dimension rewards are not provided.
        response_mask: (bs, response_length)
        index: (bs) – group id per sample (from ``uid``).
        epsilon: Numerical stability constant.
        norm_adv_by_std_in_grpo: Whether to normalize by std in GRPO.
        config: Algorithm configuration (optional).
        non_tensor_batch: Non-tensor batch data containing per-dimension reward scores.
        batch: Batch data containing prompts, attention_mask, etc.

    Note:
        Ref GDPO (https://arxiv.org/abs/2601.05242).

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) – same as advantages (outcome-only).
    """
    score_list = None
    reward_weights = None

    if config is not None and non_tensor_batch is not None and batch is not None:
        gdpo_reward_keys = config.get("gdpo_reward_keys", None)
        assert gdpo_reward_keys, (
            "GDPO requires 'algorithm.gdpo_reward_keys' listing the individual reward "
            "component keys returned by compute_score (e.g. ['format_reward', 'accuracy_reward'])."
        )
        device = token_level_rewards.device
        prompt_length = batch["prompts"].size(1)
        valid_response_length = batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1

        score_list = []
        for key in gdpo_reward_keys:
            assert key in non_tensor_batch, (
                f"GDPO reward key '{key}' not found in non_tensor_batch. "
                f"Available keys: {list(non_tensor_batch.keys())}. "
                f"Make sure your compute_score returns a dict containing '{key}'."
            )
            comp = non_tensor_batch[key]
            rm_score = torch.tensor(np.asarray(comp, dtype=np.float32), device=device)
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(rm_scores.size(0), device=device), valid_response_length] = rm_score
            score_list.append(rm_scores)

        gdpo_weights = config.get("gdpo_reward_weights", None)
        if gdpo_weights is not None:
            reward_weights = list(gdpo_weights)

    if score_list is None:
        score_list = [token_level_rewards]

    num_scores = len(score_list)

    if reward_weights is not None:
        weights = torch.tensor(reward_weights, dtype=torch.float32, device=token_level_rewards.device)
    else:
        weights = torch.ones(num_scores, dtype=torch.float32, device=token_level_rewards.device)

    new_advantage = None

    for i in range(num_scores):
        normalized_score, _ = compute_grpo_outcome_advantage(
            token_level_rewards=score_list[i],
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config)

        if new_advantage is None:
            new_advantage = weights[i] * normalized_score
        else:
            new_advantage += weights[i] * normalized_score

    advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask

    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: True
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n)]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                horizon = loss_mask.shape[-1]
                loss_scale_factor = horizon
            loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-TV.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the TV divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-TV
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    valid_positive_mask = (prob - old_prob) <= clip_divergence_high
    valid_negative_mask = (prob - old_prob) >= -clip_divergence_low
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-KL.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the KL divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-KL
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    valid_positive_mask = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
    valid_negative_mask = (binary_kl <= clip_divergence_low) | (prob >= old_prob)
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard DPPO)
    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3
    estimator is not the expected gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info)

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item()}

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold)
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights)

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics


# ================================================================== #
# Self-distillation losses (OPSD / SDPO / RLSD)
# Algorithm params are read from config.policy_loss (set via Hydra overrides)
# ================================================================== #

def _sd_token_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)


def _sd_cfg(config, key, default):
    """Read algorithm param from config.policy_loss."""
    if config is None:
        return default
    pl = getattr(config, 'policy_loss', None)
    if pl is None:
        return default
    if hasattr(pl, 'get'):
        value = pl.get(key, default)
    else:
        value = getattr(pl, key, default)
    return default if value is None else value


def _sd_teacher_metrics(teacher_lp, log_prob, response_mask, teacher_entropy=None):
    """Common teacher metrics for all self-distillation losses."""
    metrics = {}
    if teacher_lp is not None:
        metrics["teacher_gap"] = _sd_token_mean((teacher_lp - log_prob).abs(), response_mask).item()
        metrics["teacher_avg_logprob"] = _sd_token_mean(teacher_lp, response_mask).item()
        metrics["student_avg_logprob"] = _sd_token_mean(log_prob, response_mask).item()
    if teacher_entropy is not None:
        metrics["teacher_entropy"] = _sd_token_mean(teacher_entropy, response_mask).item()
    return metrics


def _sd_bool_cfg(config, key, default):
    return bool(_sd_cfg(config, key, default))


def _sd_int_cfg(config, key, default):
    return int(_sd_cfg(config, key, default))


def _sd_group_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (values * weights).sum(dim=1) / denom


def _sd_group_var(values: torch.Tensor, valid_mask: torch.Tensor, mean: torch.Tensor | None = None) -> torch.Tensor:
    if mean is None:
        mean = _sd_group_mean(values, valid_mask)
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    centered = values - mean.unsqueeze(1)
    return (centered.square() * weights).sum(dim=1) / denom


def _sd_distribution_metrics(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    prefix: str,
    include_percentiles: bool = True,
) -> dict[str, float]:
    valid_mask = response_mask.bool() & torch.isfinite(values)
    valid_values = torch.masked_select(values.float(), valid_mask)
    if valid_values.numel() == 0:
        base = {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan")}
        if include_percentiles:
            base.update({
                f"{prefix}_p05": float("nan"),
                f"{prefix}_p25": float("nan"),
                f"{prefix}_p50": float("nan"),
                f"{prefix}_p75": float("nan"),
                f"{prefix}_p95": float("nan")})
        return base

    base = {
        f"{prefix}_mean": valid_values.mean().item(),
        f"{prefix}_min": valid_values.min().item(),
        f"{prefix}_max": valid_values.max().item()}
    if not include_percentiles:
        return base
    quantiles = torch.quantile(
        valid_values,
        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=valid_values.device))
    p05, p25, p50, p75, p95 = [quantile.item() for quantile in quantiles]
    base.update({
        f"{prefix}_p05": p05,
        f"{prefix}_p25": p25,
        f"{prefix}_p50": p50,
        f"{prefix}_p75": p75,
        f"{prefix}_p95": p95})
    return base


def _sd_effective_lam(config, global_steps, default=0.5):
    """Linearly decay lambda over lam_decay_steps, keeping step 1 at the initial value."""
    lam = float(_sd_cfg(config, "lam", default))
    lam_decay_steps = int(_sd_cfg(config, "lam_decay_steps", 0) or 0)
    if lam_decay_steps <= 0 or global_steps is None:
        return lam

    step_index = max(int(global_steps) - 1, 0)
    decay_progress = min(step_index / lam_decay_steps, 1.0)
    return max(lam * (1.0 - decay_progress), 0.0)


def _sd_prepare_distill_log_probs(
    student_all_log_probs,
    teacher_all_log_probs,
    top_k=0,
    add_tail=True,
    top_k_source="student"):
    """Prepare aligned student/teacher log-probs for full-vocab or top-k distillation."""
    assert student_all_log_probs is not None and teacher_all_log_probs is not None, (
        "full_logit_distill=True requires student_all_log_probs and teacher_all_log_probs."
    )

    if top_k <= 0:
        return student_all_log_probs, teacher_all_log_probs

    top_k = min(top_k, student_all_log_probs.shape[-1])
    source_log_probs = student_all_log_probs if top_k_source == "student" else teacher_all_log_probs
    _, topk_indices = torch.topk(source_log_probs.detach(), k=top_k, dim=-1)
    student_topk = torch.gather(student_all_log_probs, dim=-1, index=topk_indices)
    teacher_topk = torch.gather(teacher_all_log_probs, dim=-1, index=topk_indices)

    if add_tail:
        def _tail_log_prob(log_probs):
            log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_mass = torch.clamp(log_mass, max=-1e-7)
            return torch.log(-torch.expm1(log_mass))

        student_tail = _tail_log_prob(student_topk)
        teacher_tail = _tail_log_prob(teacher_topk)
        student_topk = torch.cat([student_topk, student_tail], dim=-1)
        teacher_topk = torch.cat([teacher_topk, teacher_tail], dim=-1)
    else:
        student_topk = student_topk - torch.logsumexp(student_topk, dim=-1, keepdim=True)
        teacher_topk = teacher_topk - torch.logsumexp(teacher_topk, dim=-1, keepdim=True)

    return student_topk, teacher_topk


def _sd_prepare_sparse_topk_log_probs(student_topk_log_probs, teacher_topk_log_probs, add_tail=True):
    """Finalize aligned sparse top-k log-probs by adding a tail bucket or renormalizing."""
    assert student_topk_log_probs is not None and teacher_topk_log_probs is not None, (
        "Sparse top-k distillation requires student_topk_log_probs and teacher_topk_log_probs."
    )

    if add_tail:
        def _tail_log_prob(log_probs):
            log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_mass = torch.clamp(log_mass, max=-1e-7)
            return torch.log(-torch.expm1(log_mass))

        student_tail = _tail_log_prob(student_topk_log_probs)
        teacher_tail = _tail_log_prob(teacher_topk_log_probs)
        student_topk_log_probs = torch.cat([student_topk_log_probs, student_tail], dim=-1)
        teacher_topk_log_probs = torch.cat([teacher_topk_log_probs, teacher_tail], dim=-1)
    else:
        student_topk_log_probs = student_topk_log_probs - torch.logsumexp(
            student_topk_log_probs, dim=-1, keepdim=True
        )
        teacher_topk_log_probs = teacher_topk_log_probs - torch.logsumexp(
            teacher_topk_log_probs, dim=-1, keepdim=True
        )

    return student_topk_log_probs, teacher_topk_log_probs


def _sd_goldside_js_from_aligned_topk(
    gold_topk_log_probs,
    counterfactual_topk_log_probs,
    add_tail=True,
):
    """Gold-weighted JS contribution on aligned top-k support plus one tail."""
    assert (
        gold_topk_log_probs is not None
        and counterfactual_topk_log_probs is not None
    ), "gold-side JS requires both teacher top-k tensors"
    if gold_topk_log_probs.shape != counterfactual_topk_log_probs.shape:
        raise AssertionError(
            "gold/counterfactual top-k log-prob shapes must match"
        )

    gold_log_probs, counterfactual_log_probs = (
        _sd_prepare_sparse_topk_log_probs(
            student_topk_log_probs=gold_topk_log_probs.float(),
            teacher_topk_log_probs=counterfactual_topk_log_probs.float(),
            add_tail=add_tail,
        )
    )
    mixture_log_probs = torch.logaddexp(
        gold_log_probs, counterfactual_log_probs
    ) - math.log(2.0)
    goldside_js = F.kl_div(
        mixture_log_probs,
        gold_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    return (goldside_js / math.log(2.0)).clamp(min=0.0, max=1.0)


def _sd_generalized_jsd(student_log_probs, teacher_log_probs, mix, vocab_clip=None):
    """Generalized JSD / KL over log-prob distributions.

    If vocab_clip is a positive float, the per-(token, vocab) divergence
    contribution is clamped at vocab_clip BEFORE summing over the vocab
    dimension. This matches the official OPSD reference implementation, where
    `F.kl_div(..., reduction="none")` (shape [B, S, V]) is clipped per element
    and then summed across both vocab and tokens. With vocab_clip=None (the
    default), behavior is unchanged: KL contributions are summed over the
    vocab dimension first and any clipping is left to the caller.
    """
    apply_vocab_clip = vocab_clip is not None and vocab_clip > 0

    if mix == 0.0:
        per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        if apply_vocab_clip:
            per_vocab = per_vocab.clamp(max=vocab_clip)
        return per_vocab.sum(-1)
    if mix == 1.0:
        per_vocab = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        if apply_vocab_clip:
            per_vocab = per_vocab.clamp(max=vocab_clip)
        return per_vocab.sum(-1)

    mix_tensor = torch.tensor(mix, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log1p(-mix_tensor),
                teacher_log_probs + torch.log(mix_tensor),
            ]
        ),
        dim=0)
    if apply_vocab_clip:
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        per_vocab = torch.lerp(kl_student, kl_teacher, mix_tensor)
        return per_vocab.clamp(max=vocab_clip).sum(-1)
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(-1)
    return torch.lerp(kl_student, kl_teacher, mix_tensor)


def _sd_expand_sample_mask(sample_mask, response_mask, name):
    if sample_mask is None:
        raise AssertionError(f"{name} is required but missing from the actor batch")
    if sample_mask.dim() == 1:
        sample_mask = sample_mask.unsqueeze(-1)
    if sample_mask.shape[0] != response_mask.shape[0]:
        raise AssertionError(f"{name} batch dimension does not match response_mask")
    if sample_mask.shape[-1] == 1:
        sample_mask = sample_mask.expand_as(response_mask)
    elif sample_mask.shape != response_mask.shape:
        raise AssertionError(f"{name} shape does not match response_mask")
    return (
        sample_mask.to(device=response_mask.device, dtype=torch.bool)
        & response_mask.to(dtype=torch.bool)
    ).to(dtype=response_mask.dtype)


_SD_FWD_MODE_CODES = {"none": 0.0, "soft": 1.0, "const": 2.0, "hard_ce": 3.0}


def _sd_forward_pull_term(
    student_log_probs,
    teacher_log_probs,
    mode,
    teacher_topp,
    hard_threshold,
):
    """Per-token forward pull toward the (sharpened) teacher on the aligned
    top-k support, 2026-08-31.

    Motivation: the reverse-KL logit gradient carries a factor of the
    student's own probability, so tokens the student has effectively ruled
    out (p_S(teacher argmax) ~ 0) receive almost no push. On the frozen
    step-21 probe corpus the measured pull at those stuck positions was
    0.073 for reverse KL vs 0.68/0.34/0.99 for the soft/const/hard_ce
    variants (logs/20260830_entropy_decomposition, offline pull probe).
    Forward-KL-style terms have d(loss)/d(p) ~ -1/p, which cancels the
    softmax's p factor and keeps the pull alive at p ~ 0.

    Inputs are aligned [B, S, K+1] log-probs (teacher's top-k indices plus
    one tail bucket; the tail is excluded from the pull target). Student
    values must be TRUE log-probs at those indices (distill_add_tail=True
    path), so p_S(argmax) is the student's actual probability mass.

    Modes:
      soft    -- forward KL onto the top-p-sharpened teacher; caller weights
                 it by (1 - p_S(y*)), so it dies off as the student learns.
      const   -- same target, constant weight (caller applies its coef).
      hard_ce -- plain CE on the teacher argmax token, weight
                 1[p_S(y*) < hard_threshold]; no sharpening involved.

    Returns (per_token_fwd, weight, p_top1), all [B, S]; weight and p_top1
    are detached, per_token_fwd carries the student gradient.
    """
    t_lp = teacher_log_probs.detach().float()[..., :-1]
    s_lp = student_log_probs.float()[..., :-1]
    top1 = t_lp.argmax(dim=-1, keepdim=True)
    s_at_top1 = s_lp.gather(-1, top1).squeeze(-1)
    p_top1 = s_at_top1.detach().exp()

    if mode == "hard_ce":
        per_token = -s_at_top1
        weight = (p_top1 < hard_threshold).to(per_token.dtype)
        return per_token, weight, p_top1

    # Top-p sharpen the teacher: keep the smallest prefix of tokens (by
    # descending probability) whose cumulative mass reaches teacher_topp,
    # then renormalize. This bounds the target's entropy so the pull never
    # drags the student toward the teacher's flat tail (the 20260824
    # entropy-inflation channel).
    sorted_lp, order = t_lp.sort(dim=-1, descending=True)
    sorted_p = sorted_lp.exp()
    cumulative = sorted_p.cumsum(-1)
    keep_sorted = (cumulative - sorted_p) < teacher_topp
    keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
    kept_p = t_lp.exp() * keep
    mass = kept_p.sum(-1, keepdim=True).clamp_min(1e-9)
    q = kept_p / mass
    log_q = torch.where(keep, t_lp - mass.log(), torch.zeros_like(t_lp))
    per_token = (q * (log_q - s_lp)).sum(-1)

    if mode == "soft":
        weight = (1.0 - p_top1)
    elif mode == "const":
        weight = torch.ones_like(p_top1)
    else:
        raise ValueError(f"unknown distill_fwd_mode {mode!r}")
    return per_token, weight.detach(), p_top1


def _distill_eligibility_mask(response_mask, config, kwargs):
    """Tokens the distill term is averaged over, honouring ``rlsd_skip_math``.

    The distill family has no advantage to fall back on, so "skip math" has to
    mean "these rows contribute no distill gradient". Math rows in this mix carry
    no worked solution, so their privileged view degenerates to
    math_privileged_text_mode=answer_only -- a teacher that has been handed the
    answer -- and distilling onto it teaches the student to guess toward it. The
    20260812 OPSD arm dropped math from the DATA for exactly this reason; masking
    it here keeps the dataset (and therefore the step count and the comparison
    set) identical instead.

    The mean is taken over ELIGIBLE tokens, not all response tokens, so the
    per-token distill magnitude -- and hence what distill_coef means -- does not
    move when the math fraction of the mix changes. Returns (mask, ratio) where
    ratio is NaN when the flag is off, since metric keys must be constant.
    """
    if _sd_bool_cfg(config, "distill_key_ablation_gate", False) or _sd_bool_cfg(config, "distill_certified_only", False):
        keyabl_mask = _sd_expand_sample_mask(
            kwargs.get("distill_key_ablation_sample_mask"),
            response_mask,
            "distill_key_ablation_sample_mask",
        )
        return keyabl_mask, _sd_token_mean(keyabl_mask, response_mask).item()

    if not _sd_bool_cfg(config, "rlsd_skip_math", False):
        return response_mask, float("nan")
    evidence_sample_mask = kwargs.get("evidence_sample_mask")
    assert evidence_sample_mask is not None, (
        "rlsd_skip_math=True requires evidence_sample_mask in the batch; check "
        "_compute_teacher_log_probs attached it for this loss_mode."
    )
    gate = _sd_expand_sample_mask(
        evidence_sample_mask, response_mask, "evidence_sample_mask"
    )
    ratio = _sd_token_mean(gate, response_mask).item()
    return gate, ratio


def _gate_matching_counts(hard_gate, eligible_mask, priority):
    """Select the lowest K priority values within each rollout, breaking ties by position."""
    eligible = eligible_mask.bool()
    counts = (hard_gate.bool() & eligible).sum(dim=-1, keepdim=True)
    order = priority.detach().masked_fill(~eligible, float("inf")).argsort(dim=-1, stable=True)
    first_k = torch.arange(order.size(-1), device=order.device) < counts
    return torch.zeros_like(eligible).scatter_(-1, order, first_k)


def _random_gate_matching_counts(hard_gate, eligible_mask):
    """Uniformly sample K eligible positions per rollout, including K=0."""
    keys = torch.rand(eligible_mask.shape, device=eligible_mask.device, dtype=torch.float64)
    return _gate_matching_counts(hard_gate, eligible_mask, keys)


def _opsd_distill_term(log_prob, response_mask, config, rollout_is_weights, **kwargs):
    """The OPSD distillation loss and its metrics, with no policy-gradient term.

    Factored out of ``compute_policy_loss_opsd`` so ``grpo_opsd`` can add the
    same term to a vanilla PG loss instead of replacing it. Behavior for
    ``loss_mode=opsd`` is unchanged: that mode still returns exactly this loss.
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "OPSD requires teacher_log_probs but got None. "
        "Ensure teacher_mode=fixed and use_kl_loss=True so ref_log_prob is computed."
    )
    beta = float(_sd_cfg(config, "beta", 0.0))
    jsd_token_clip = float(_sd_cfg(config, "jsd_token_clip", 0.05))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="teacher")
        # Match official OPSD: clip per-(token, vocab) KL contribution before
        # summing over vocab. The clip can break Gibbs' inequality and produce
        # negative per-token loss values; this matches the reference behavior.
        per_token_loss = _sd_generalized_jsd(
            student_distill,
            teacher_distill,
            beta,
            vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)
    else:
        per_token_loss = (teacher_lp.detach() - log_prob).pow(2)
        if jsd_token_clip > 0:
            per_token_loss = per_token_loss.clamp(max=jsd_token_clip)

    gate_mode = str(_sd_cfg(config, "distill_key_ablation_gate_mode", "sampled_negative_soft"))
    if gate_mode == "goldside_js_kl_matched":
        if not full_logit_distill or beta != 1.0 or jsd_token_clip > 0:
            raise ValueError("KL-ranked selection requires full_logit_distill, beta=1 and unclipped KL")
        # Rank by the same reverse KL used for distillation, before optional loss weights.
        token_reverse_kl = per_token_loss.detach()

    # ---- Forward-pull term (2026-08-31, additive; "none" is byte-identical
    # to the 20260829 parent). See _sd_forward_pull_term for the mechanism.
    fwd_mode = str(_sd_cfg(config, "distill_fwd_mode", "none"))
    if fwd_mode not in _SD_FWD_MODE_CODES:
        raise ValueError(f"unknown distill_fwd_mode {fwd_mode!r}")
    fwd_coef = float(_sd_cfg(config, "distill_fwd_coef", 1.0))
    fwd_topp = float(_sd_cfg(config, "distill_fwd_teacher_topp", 0.9))
    fwd_hard_threshold = float(
        _sd_cfg(config, "distill_fwd_hard_threshold", 0.1)
    )
    fwd_contrib = None
    fwd_weight = None
    fwd_p_top1 = None
    if fwd_mode != "none":
        if not math.isfinite(fwd_coef) or fwd_coef <= 0:
            raise ValueError(f"distill_fwd_coef must be positive, got {fwd_coef}")
        if not 0.0 < fwd_topp <= 1.0:
            raise ValueError(
                f"distill_fwd_teacher_topp must be in (0, 1], got {fwd_topp}"
            )
        assert full_logit_distill and top_k_distill > 0 and distill_add_tail, (
            "distill_fwd_mode requires full_logit_distill=True, "
            "top_k_distill>0 and distill_add_tail=True so student log-probs "
            "at the teacher's top-k indices are true (unrenormalized) "
            "probabilities."
        )
        per_token_fwd, fwd_weight, fwd_p_top1 = _sd_forward_pull_term(
            student_distill,
            teacher_distill,
            fwd_mode,
            fwd_topp,
            fwd_hard_threshold,
        )
        fwd_contrib = (
            fwd_coef * fwd_weight * per_token_fwd
        ).to(per_token_loss.dtype)
        per_token_loss = per_token_loss + fwd_contrib

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    distill_mask, eligible_ratio = _distill_eligibility_mask(response_mask, config, kwargs)

    keyabl_gate_enabled = _sd_bool_cfg(
        config, "distill_key_ablation_gate", False
    )
    keyabl_gate_token_denominator = _sd_bool_cfg(
        config, "distill_key_ablation_gate_token_denominator", False
    )
    if keyabl_gate_token_denominator:
        if not keyabl_gate_enabled:
            raise ValueError(
                "distill_key_ablation_gate_token_denominator=True requires "
                "distill_key_ablation_gate=True"
            )
        requested_gate_mode = str(_sd_cfg(
            config,
            "distill_key_ablation_gate_mode",
            "sampled_negative_soft",
        ))
        if requested_gate_mode != "goldside_js_hard":
            raise ValueError(
                "distill_key_ablation_gate_token_denominator=True requires "
                "distill_key_ablation_gate_mode='goldside_js_hard'"
            )
        if _sd_bool_cfg(config, "distill_wrong_only", False):
            raise ValueError(
                "active-gate-token denominator is only defined for the "
                "symmetric ALL arm, not distill_wrong_only"
            )
        if float(_sd_cfg(config, "distill_coef_correct", -1.0)) >= 0.0:
            raise ValueError(
                "active-gate-token denominator is only defined for the "
                "symmetric ALL arm, not distill_coef_correct"
            )
    loss_denominator_mask = distill_mask
    loss_denominator_token_ratio = 1.0
    keyabl_gate_mean = float("nan")
    keyabl_gate_active_ratio = float("nan")
    keyabl_gate_saturated_ratio = float("nan")
    keyabl_shift_mean = float("nan")
    keyabl_shift_min = float("nan")
    keyabl_shift_max = float("nan")
    keyabl_goldjs_mean = float("nan")
    keyabl_goldjs_min = float("nan")
    keyabl_goldjs_max = float("nan")
    keyabl_goldjs_p50 = float("nan")
    keyabl_goldjs_p90 = float("nan")
    keyabl_goldjs_p95 = float("nan")
    keyabl_goldjs_p99 = float("nan")
    keyabl_gold_topk_mass_mean = float("nan")
    keyabl_counterfactual_on_gold_topk_mass_mean = float("nan")
    if keyabl_gate_enabled:
        teacher_keyabl_lp = kwargs.get("teacher_keyabl_log_probs")
        assert teacher_keyabl_lp is not None, (
            "distill_key_ablation_gate=True requires "
            "teacher_keyabl_log_probs from the snapshot teacher forward."
        )
        if (
            teacher_lp.shape != response_mask.shape
            or teacher_keyabl_lp.shape != response_mask.shape
        ):
            raise AssertionError(
                "gold/key-ablated sampled log-prob shapes must match response_mask"
            )
        if not torch.isfinite(teacher_lp).all():
            raise ValueError("gold teacher sampled log probabilities are non-finite")
        if not torch.isfinite(teacher_keyabl_lp).all():
            raise ValueError(
                "key-ablated teacher sampled log probabilities are non-finite"
            )

        tau = float(_sd_cfg(config, "distill_key_ablation_tau", 0.1))
        sampled_shift = (
            teacher_lp.detach() - teacher_keyabl_lp.detach()
        )
        if not torch.isfinite(sampled_shift).all():
            raise ValueError("key-ablation sampled-token shifts are non-finite")

        if gate_mode == "sampled_negative_soft":
            saturation = float(_sd_cfg(
                config, "distill_key_ablation_saturation", 0.6
            ))
            if not math.isfinite(tau) or not math.isfinite(saturation):
                raise ValueError("key-ablation gate thresholds must be finite")
            if tau < 0.0 or saturation <= tau:
                raise ValueError(
                    "key-ablation gate requires 0 <= tau < saturation; "
                    f"got tau={tau}, saturation={saturation}"
                )
            gate_margin = -sampled_shift - tau
            gate_width = saturation - tau
            boundary_tol = 8.0 * torch.finfo(gate_margin.dtype).eps
            interpolated_gate = torch.clamp(
                gate_margin / gate_width, min=0.0, max=1.0
            )
            keyabl_gate = torch.where(
                gate_margin <= boundary_tol,
                torch.zeros_like(interpolated_gate),
                torch.where(
                    gate_margin >= gate_width - boundary_tol,
                    torch.ones_like(interpolated_gate),
                    interpolated_gate,
                ),
            ).detach()
        elif gate_mode in ("goldside_js_hard", "goldside_js_random_matched", "goldside_js_kl_matched"):
            if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
                raise ValueError(
                    "gold-side JS hard gate requires 0 <= tau <= 1; "
                    f"got tau={tau}"
                )
            teacher_topk = kwargs.get("teacher_topk_log_probs")
            teacher_keyabl_topk = kwargs.get(
                "teacher_keyabl_topk_log_probs"
            )
            assert teacher_topk is not None and teacher_keyabl_topk is not None, (
                "gold-side JS hard gate requires aligned gold and "
                "counterfactual teacher top-k log probabilities"
            )
            goldside_js = _sd_goldside_js_from_aligned_topk(
                gold_topk_log_probs=teacher_topk.detach(),
                counterfactual_topk_log_probs=teacher_keyabl_topk.detach(),
                add_tail=_sd_bool_cfg(config, "distill_add_tail", True),
            )
            if goldside_js.shape != response_mask.shape:
                raise AssertionError(
                    "gold-side JS shape must match response_mask"
                )
            if not torch.isfinite(goldside_js).all():
                raise ValueError("gold-side JS scores are non-finite")
            keyabl_gate = (goldside_js > tau).to(
                dtype=per_token_loss.dtype
            ).detach()
            if gate_mode == "goldside_js_random_matched":
                keyabl_gate = _random_gate_matching_counts(keyabl_gate, distill_mask).to(per_token_loss.dtype)
            elif gate_mode == "goldside_js_kl_matched":
                keyabl_gate = _gate_matching_counts(keyabl_gate, distill_mask, -token_reverse_kl).to(per_token_loss.dtype)

            eligible_goldjs = torch.masked_select(
                goldside_js.float(), distill_mask.to(dtype=torch.bool)
            )
            if eligible_goldjs.numel() > 0:
                keyabl_goldjs_mean = eligible_goldjs.mean().item()
                keyabl_goldjs_min = eligible_goldjs.min().item()
                keyabl_goldjs_max = eligible_goldjs.max().item()
                quantiles = torch.quantile(
                    eligible_goldjs,
                    torch.tensor(
                        [0.50, 0.90, 0.95, 0.99],
                        device=eligible_goldjs.device,
                    ),
                )
                (
                    keyabl_goldjs_p50,
                    keyabl_goldjs_p90,
                    keyabl_goldjs_p95,
                    keyabl_goldjs_p99,
                ) = [value.item() for value in quantiles]

                gold_topk_mass = teacher_topk.detach().float().logsumexp(
                    dim=-1
                ).exp()
                counterfactual_on_gold_topk_mass = (
                    teacher_keyabl_topk.detach().float().logsumexp(
                        dim=-1
                    ).exp()
                )
                eligible = distill_mask.to(dtype=torch.bool)
                keyabl_gold_topk_mass_mean = torch.masked_select(
                    gold_topk_mass, eligible
                ).mean().item()
                keyabl_counterfactual_on_gold_topk_mass_mean = (
                    torch.masked_select(
                        counterfactual_on_gold_topk_mass, eligible
                    ).mean().item()
                )
        else:
            raise ValueError(
                "unknown distill_key_ablation_gate_mode "
                f"{gate_mode!r}"
            )

        per_token_loss = per_token_loss * keyabl_gate
        if keyabl_gate_token_denominator:
            loss_denominator_mask = distill_mask * (keyabl_gate > 0).to(
                dtype=distill_mask.dtype
            )
            loss_denominator_token_ratio = _sd_token_mean(
                loss_denominator_mask, distill_mask
            ).item()

        keyabl_gate_mean = _sd_token_mean(keyabl_gate, distill_mask).item()
        keyabl_gate_active_ratio = _sd_token_mean(
            (keyabl_gate > 0).to(dtype=distill_mask.dtype), distill_mask
        ).item()
        keyabl_gate_saturated_ratio = _sd_token_mean(
            (keyabl_gate >= 1).to(dtype=distill_mask.dtype), distill_mask
        ).item()
        eligible_shifts = torch.masked_select(
            sampled_shift.float(), distill_mask.to(dtype=torch.bool)
        )
        if eligible_shifts.numel() > 0:
            keyabl_shift_mean = eligible_shifts.mean().item()
            keyabl_shift_min = eligible_shifts.min().item()
            keyabl_shift_max = eligible_shifts.max().item()

    fwd_weighted_mean = float("nan")
    fwd_weight_mean = float("nan")
    fwd_ps_top1_mean = float("nan")
    if fwd_contrib is not None:
        if keyabl_gate_enabled:
            fwd_gated = fwd_contrib * keyabl_gate
            gate_mask = distill_mask * keyabl_gate
        else:
            fwd_gated = fwd_contrib
            gate_mask = distill_mask
        # Same denominator convention as distill_loss, including the
        # optional active-hard-gate-token denominator.
        fwd_weighted_mean = _sd_token_mean(
            fwd_gated, loss_denominator_mask
        ).item()
        # Weight and student-mass-on-teacher-argmax are averaged over the
        # GATED eligible tokens: that is the population the term acts on and
        # matches the stuck/hesitating decomposition of the 20260830 probe.
        # NOTE: keep these float32. response_mask (and therefore
        # distill_mask) arrives as int64 from dp_actor, and casting the
        # weights to the mask dtype truncated 0.93 -> 0 in the first SDB
        # smoke round (2026-08-31), degenerating both metrics into
        # saturated-position counters.
        fwd_weight_mean = _sd_token_mean(
            fwd_weight.float(), gate_mask.float()
        ).item()
        fwd_ps_top1_mean = _sd_token_mean(
            fwd_p_top1.float(), gate_mask.float()
        ).item()

    wrong_only = _sd_bool_cfg(config, "distill_wrong_only", False)
    wrong_token_ratio = float("nan")
    wrong_conditional_loss = float("nan")
    right_conditional_loss = float("nan")
    # ---- Asymmetric per-side coefficient (2026-08-31). distill_coef keeps
    # its wrong-side meaning; distill_coef_correct >= 0 sets the coefficient
    # on reward-correct trajectories (implemented as a relative weight, since
    # grpo_opsd multiplies the whole term by distill_coef afterwards).
    # Motivation (gradient_conflict_probe, calibrated on the 7 evaluated
    # arms): reverse-KL's PG-conflict and entropy pressure concentrate on
    # CORRECT trajectories where the teacher disagrees with a winning sample
    # (0.56/1.87 per token at right/stuck vs 0.04/? at wrong/stuck), while
    # rescue-at-hesitating -- the only probe currency that tracks eval --
    # is minted on both sides. Big mu on the wrong side buys rescue without
    # the conflict tax; small mu on the correct side keeps the consolidation
    # value that the wrong-only arm (61.87 vs 63.10) proved real.
    coef_correct = float(_sd_cfg(config, "distill_coef_correct", -1.0))
    asym = coef_correct >= 0.0
    if asym:
        assert not wrong_only, (
            "distill_coef_correct and distill_wrong_only are mutually "
            "exclusive; wrong_only is the coef_correct=0 special case."
        )
        distill_coef_wrong = float(_sd_cfg(config, "distill_coef", 0.0))
        assert distill_coef_wrong > 0.0, (
            "distill_coef_correct requires distill_coef > 0 (the wrong-side "
            "coefficient that the relative weight is expressed against)."
        )
        distill_sample_mask = kwargs.get("distill_sample_mask")
        assert distill_sample_mask is not None, (
            "distill_coef_correct>=0 requires distill_sample_mask in the "
            "actor batch; check the reward-mask plumbing in "
            "self_distill_main.py."
        )
        wrong_token_mask = _sd_expand_sample_mask(
            distill_sample_mask, response_mask, "distill_sample_mask"
        ) * distill_mask
        right_token_mask = (distill_mask - wrong_token_mask).clamp(min=0.0)
        ratio = coef_correct / distill_coef_wrong
        weight = wrong_token_mask + ratio * right_token_mask
        wrong_token_ratio = _sd_token_mean(wrong_token_mask, distill_mask).item()
        wrong_conditional_loss = _sd_token_mean(
            per_token_loss, wrong_token_mask
        ).item()
        right_conditional_loss = _sd_token_mean(
            per_token_loss, right_token_mask
        ).item()
        # All-eligible denominator, same convention as wrong_only: the
        # wrong-side dose self-anneals as accuracy improves.
        loss = _sd_token_mean(per_token_loss * weight, distill_mask)
    elif wrong_only:
        distill_sample_mask = kwargs.get("distill_sample_mask")
        assert distill_sample_mask is not None, (
            "distill_wrong_only=True requires distill_sample_mask in the actor batch; "
            "check _compute_teacher_log_probs reward-mask plumbing."
        )
        wrong_token_mask = _sd_expand_sample_mask(
            distill_sample_mask, response_mask, "distill_sample_mask"
        ) * distill_mask
        wrong_token_ratio = _sd_token_mean(wrong_token_mask, distill_mask).item()
        wrong_conditional_loss = _sd_token_mean(
            per_token_loss, wrong_token_mask
        ).item()
        # Keep the all-eligible denominator. Correct rows contribute zero to
        # the numerator, making the auxiliary loss self-anneal with accuracy.
        loss = _sd_token_mean(per_token_loss * wrong_token_mask, distill_mask)
    else:
        loss = _sd_token_mean(per_token_loss, loss_denominator_mask)
    metrics = {
        "distill_loss": loss.item(),
        "distill_beta": beta,
        "distill_topk": float(top_k_distill),
        "distill_eligible_token_ratio": eligible_ratio,
        "distill_full_logit": float(full_logit_distill),
        "distill_wrong_only": float(wrong_only),
        "distill_coef_correct": coef_correct,
        "distill_wrong_token_ratio": wrong_token_ratio,
        "distill_wrong_conditional_loss": wrong_conditional_loss,
        "distill_right_conditional_loss": right_conditional_loss,
        "distill_fwd_mode_code": _SD_FWD_MODE_CODES[fwd_mode],
        "distill_fwd_coef": fwd_coef,
        "distill_fwd_weighted": fwd_weighted_mean,
        "distill_fwd_weight_mean": fwd_weight_mean,
        "distill_fwd_ps_top1_mean": fwd_ps_top1_mean,
        "distill_keyabl_gate": float(keyabl_gate_enabled),
        "distill_keyabl_gate_token_denominator": float(
            keyabl_gate_token_denominator
        ),
        "distill_loss_denominator_token_ratio": (
            loss_denominator_token_ratio
        ),
        "distill_keyabl_gate_mean": keyabl_gate_mean,
        "distill_keyabl_gate_active_ratio": keyabl_gate_active_ratio,
        "distill_keyabl_gate_saturated_ratio": keyabl_gate_saturated_ratio,
        "distill_keyabl_shift_mean": keyabl_shift_mean,
        "distill_keyabl_shift_min": keyabl_shift_min,
        "distill_keyabl_shift_max": keyabl_shift_max,
        "distill_keyabl_goldjs_mean": keyabl_goldjs_mean,
        "distill_keyabl_goldjs_min": keyabl_goldjs_min,
        "distill_keyabl_goldjs_max": keyabl_goldjs_max,
        "distill_keyabl_goldjs_p50": keyabl_goldjs_p50,
        "distill_keyabl_goldjs_p90": keyabl_goldjs_p90,
        "distill_keyabl_goldjs_p95": keyabl_goldjs_p95,
        "distill_keyabl_goldjs_p99": keyabl_goldjs_p99,
        "distill_keyabl_gold_topk_mass_mean": keyabl_gold_topk_mass_mean,
        "distill_keyabl_counterfactual_on_gold_topk_mass_mean": (
            keyabl_counterfactual_on_gold_topk_mass_mean
        )}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("opsd")
def compute_policy_loss_opsd(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """OPSD: Forward KL distillation (teacher -> student). No PG term."""
    return _opsd_distill_term(log_prob, response_mask, config, rollout_is_weights, **kwargs)


@register_policy_loss("grpo_opsd")
def compute_policy_loss_grpo_opsd(old_log_prob, log_prob, advantages, response_mask,
                                  loss_agg_mode="token_mean", config=None,
                                  rollout_is_weights=None, **kwargs):
    """GRPO policy gradient PLUS an additive OPSD distillation term.

        loss = pg_loss(advantages) + distill_coef * forward_KL(teacher -> student)

    Unlike ``opsd``, which REPLACES the policy gradient (dp_actor.py sets
    ``policy_loss = pg_loss`` from whatever the loss fn returns), this keeps the
    reward signal intact and treats distillation as an auxiliary term. The
    reward-shaped part is byte-identical to ``vanilla``, so distill_coef=0
    reproduces GRPO exactly.

    Two settings interact and must be read together:
      - ``distill_coef`` is NOT a fraction of the update. The raw distill
        gradient is far larger than the PG gradient (measured step-0 grad_norm
        on Qwen3-1.7B: 2.36 for opsd at jsd_token_clip=0.05 vs 0.40 for grpo,
        i.e. ~6x), so distill_coef ~= 0.17 is where the two terms are equal AT
        THAT CLIP.
      - ``jsd_token_clip=0`` (no per-vocab clamp) restores a true divergence,
        but the clamp was suppressing the largest contributions: removing it
        multiplies the distill gradient by roughly another 5x (see
        scripts_local/opsd_clip_entropy_probe.py). Equal weight then lands
        near distill_coef ~= 0.03, not 0.17.
    Log ``distill_over_pg`` to see where a run actually sits; it is a ratio of
    loss magnitudes, not of gradient norms, so treat it as an indicator only.
    """
    distill_coef = float(_sd_cfg(config, "distill_coef", 0.0))
    pg_loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights)
    metrics["pg_loss_component"] = pg_loss.detach().item()
    metrics["distill_coef"] = distill_coef

    if distill_coef == 0.0:
        # Identity with vanilla GRPO: skip the distill graph entirely rather
        # than multiplying it by zero, so a coef=0 arm is a true GRPO control
        # even if the teacher plumbing is misconfigured. distill_coef is fixed
        # for a run, so the key set below is still constant across micro-batches
        # (see the note in compute_policy_loss_ce_rlsd); NaN marks "not measured".
        for key in ("distill_loss", "distill_beta", "distill_topk", "distill_full_logit",
                    "distill_eligible_token_ratio", "distill_wrong_token_ratio",
                    "distill_wrong_conditional_loss",
                    "distill_right_conditional_loss",
                    "distill_loss_denominator_token_ratio",
                    "distill_keyabl_gate_mean",
                    "distill_keyabl_gate_active_ratio",
                    "distill_keyabl_gate_saturated_ratio",
                    "distill_keyabl_shift_mean",
                    "distill_keyabl_shift_min",
                    "distill_keyabl_shift_max",
                    "distill_fwd_coef", "distill_fwd_weighted",
                    "distill_fwd_weight_mean", "distill_fwd_ps_top1_mean",
                    "teacher_gap", "teacher_avg_logprob", "student_avg_logprob"):
            metrics[key] = float("nan")
        metrics["distill_fwd_mode_code"] = _SD_FWD_MODE_CODES.get(
            str(_sd_cfg(config, "distill_fwd_mode", "none")), float("nan")
        )
        metrics["distill_coef_correct"] = float(
            _sd_cfg(config, "distill_coef_correct", -1.0)
        )
        metrics["distill_wrong_only"] = float(
            _sd_bool_cfg(config, "distill_wrong_only", False)
        )
        metrics["distill_keyabl_gate"] = float(
            _sd_bool_cfg(config, "distill_key_ablation_gate", False)
        )
        metrics["distill_keyabl_gate_token_denominator"] = float(
            _sd_bool_cfg(
                config,
                "distill_key_ablation_gate_token_denominator",
                False,
            )
        )
        metrics["distill_weighted"] = 0.0
        metrics["distill_over_pg"] = 0.0
        return pg_loss, metrics

    distill_loss, distill_metrics = _opsd_distill_term(
        log_prob, response_mask, config, rollout_is_weights, **kwargs)
    metrics.update(distill_metrics)
    weighted = distill_coef * distill_loss
    metrics["distill_weighted"] = weighted.detach().item()
    metrics["distill_over_pg"] = abs(metrics["distill_weighted"]) / (abs(metrics["pg_loss_component"]) + 1e-8)
    return pg_loss + weighted, metrics


def _opsd_ectr_distill_term(log_prob, response_mask, config, rollout_is_weights, **kwargs):
    """OPSD with CFG-style contrastive guided target in the top-k subspace.

    Same sparse top-k machinery as OPSD: top-k vocab indices are chosen by the
    correct-context teacher, and student / correct-teacher / wrong-teacher all
    gather their log-probs at those indices. An optional tail bucket captures
    the remaining mass. In this (top-k + tail) log-space:

        guided_logits = (1 + α) · log p_c  -  α · clamp(log p_w, min=floor)
        q_target      = softmax(guided_logits)

    Student matches q_target via the same generalized-JSD machinery as OPSD.
    α = 0 recovers vanilla OPSD; α > 0 amplifies the correct-vs-wrong contrast
    across the top-k support.

    Token-level contrastive masking (controlled by ``opsd_ectr_cw_tau``):
    At each token position the KL divergence between correct and wrong teachers
    is computed.  Positions where correct ≈ wrong (stylistic tokens) receive
    near-zero loss weight so the student's policy is left unchanged there;
    positions where the teachers genuinely disagree (content tokens) receive
    full loss weight.  tau controls the sharpness of this soft mask —
    setting tau=0 disables masking and recovers the original uniform behaviour.
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    student_topk = kwargs.get("student_topk_log_probs")
    teacher_correct_topk = kwargs.get("teacher_topk_log_probs")
    teacher_wrong_topk = kwargs.get("teacher_wrong_topk_log_probs")
    assert (
        student_topk is not None
        and teacher_correct_topk is not None
        and teacher_wrong_topk is not None
    ), (
        "opsd_ectr requires student/teacher_correct/teacher_wrong topk_log_probs "
        "gathered at teacher_correct's top-k indices."
    )

    alpha = float(_sd_cfg(config, "opsd_ectr_alpha", 0.5))
    log_w_floor = float(_sd_cfg(config, "opsd_ectr_log_w_floor", -30.0))
    cw_tau = float(_sd_cfg(config, "opsd_ectr_cw_tau", 1.0))
    beta = float(_sd_cfg(config, "beta", 0.0))
    jsd_token_clip = float(_sd_cfg(config, "jsd_token_clip", 0.05))
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    student_dist, correct_dist = _sd_prepare_sparse_topk_log_probs(
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_correct_topk,
        add_tail=distill_add_tail)
    _, wrong_dist = _sd_prepare_sparse_topk_log_probs(
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_wrong_topk,
        add_tail=distill_add_tail)

    wrong_clamped = wrong_dist.clamp(min=log_w_floor)
    guided_logits = (1.0 + alpha) * correct_dist - alpha * wrong_clamped
    guided_log_probs = guided_logits - torch.logsumexp(guided_logits, dim=-1, keepdim=True)

    per_token_loss = _sd_generalized_jsd(
        student_dist,
        guided_log_probs,
        beta,
        vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)

    # Token-level contrastive mask: only train where correct and wrong teachers
    # genuinely disagree; leave the student's policy untouched at stylistic
    # positions where the hint carries no information.
    if cw_tau > 0:
        with torch.no_grad():
            cw_kl = F.kl_div(
                wrong_dist, correct_dist, reduction="none", log_target=True).sum(-1)  # [B, S]
            cw_kl = cw_kl.clamp(min=0.0)
            token_weight = cw_kl / (cw_kl + cw_tau)
        per_token_loss = per_token_loss * token_weight

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    distill_mask, eligible_ratio = _distill_eligibility_mask(response_mask, config, kwargs)
    loss = _sd_token_mean(per_token_loss, distill_mask)

    metrics = {
        "distill_loss": loss.item(),
        "distill_beta": beta,
        "distill_topk": float(teacher_correct_topk.shape[-1]),
        "distill_eligible_token_ratio": eligible_ratio,
        "distill_full_logit": 1.0,
        "opsd_ectr_alpha": alpha,
        "opsd_ectr_log_w_floor": log_w_floor,
        "opsd_ectr_cw_tau": cw_tau}
    if cw_tau > 0:
        active = _sd_token_mean(token_weight, response_mask).item()
        metrics["opsd_ectr_active_ratio"] = active
    metrics.update(_sd_teacher_metrics(teacher_correct_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


def _opsd_ectr_metric_keys(config):
    """The exact key set ``_opsd_ectr_distill_term`` emits, for NaN-filling.

    Metrics are merged across ranks positionally, so a key present in only some
    micro-batches produces ragged per-rank lists and crashes reduce_metrics. A
    distill_coef=0 arm skips the distill graph entirely and must therefore
    declare the same keys with NaN. ``opsd_ectr_cw_tau`` is fixed for a run, so
    branching on it here still yields a constant key set.
    """
    keys = [
        "distill_loss", "distill_beta", "distill_topk", "distill_full_logit",
        "distill_eligible_token_ratio",
        "opsd_ectr_alpha", "opsd_ectr_log_w_floor", "opsd_ectr_cw_tau",
        "teacher_gap", "teacher_avg_logprob", "student_avg_logprob"]
    if float(_sd_cfg(config, "opsd_ectr_cw_tau", 1.0)) > 0:
        keys.append("opsd_ectr_active_ratio")
    return keys


@register_policy_loss("opsd_ectr")
def compute_policy_loss_opsd_ectr(old_log_prob, log_prob, advantages, response_mask,
                                  loss_agg_mode="token_mean", config=None,
                                  rollout_is_weights=None, **kwargs):
    """OPSD-ectr: the contrastive guided-target distill term, with no PG term.

    Factored exactly like ``opsd``/``grpo_opsd``: the term itself lives in
    ``_opsd_ectr_distill_term`` so ``ce_rlsd_opsd_ectr`` can ADD it to a
    CE-modulated policy gradient instead of replacing it. Behavior for
    ``loss_mode=opsd_ectr`` is unchanged -- this mode still returns exactly the
    distill loss and never sees the reward.
    """
    return _opsd_ectr_distill_term(log_prob, response_mask, config, rollout_is_weights, **kwargs)


def _sd_masked_mean(values, mask):
    """Mean of ``values`` over finite entries of a boolean/float ``mask``."""
    valid = mask.bool() & torch.isfinite(values)
    if valid.sum() == 0:
        return float("nan")
    return torch.masked_select(values.float(), valid).mean().item()


def _sd_masked_frac(condition, mask):
    """Fraction of masked positions where ``condition`` holds."""
    valid = mask.bool()
    denom = valid.sum()
    if denom == 0:
        return float("nan")
    return ((condition & valid).sum().float() / denom.float()).item()


def _rlsd_diagnostics(
    delta_t,
    teacher_lp,
    response_mask,
    epsilon_w,
    prefix,
    advantages,
    gate=None,
    gamma_t=None,
):
    """Read-only diagnostics for the RLSD/EG-RLSD evidence ratio.

    These answer three questions the aggregate metrics cannot:
      1. is ``delta_t`` two-sided, or is its sign fixed by the context gap?
      2. does the gate select tokens where the teacher is competent (small
         ``|delta_t|``) or, inverted, where it collapses (large ``|delta_t|``)?
      3. how much of the clip range is actually used, and how heavy is the
         ``gamma_t`` tail?
    """
    with torch.no_grad():
        mask = response_mask.bool()
        abs_delta = delta_t.abs()
        # Saturation is decided on the same quantity the loss clips:
        # sign(A) * delta outside [ln(1 - eps_w), ln(1 + eps_w)].
        # With eps_w >= 1 the lower clamp bound (1 - eps_w) <= 0 can never
        # bind (exp > 0), so the lower saturation threshold is -inf.
        scaled_delta = advantages.sign() * delta_t
        lower_bound = math.log(1.0 - epsilon_w) if epsilon_w < 1.0 else float("-inf")
        saturates = (scaled_delta <= lower_bound) | (
            scaled_delta >= math.log1p(epsilon_w)
        )
        metrics = {
            f"{prefix}_diag_delta_signed_mean": _sd_masked_mean(delta_t, mask),
            f"{prefix}_diag_delta_abs_mean": _sd_masked_mean(abs_delta, mask),
            f"{prefix}_diag_delta_negative_frac": _sd_masked_frac(
                delta_t < 0, mask
            ),
            f"{prefix}_diag_clip_frac": _sd_masked_frac(saturates, mask),
        }
        metrics.update(
            _sd_distribution_metrics(delta_t, response_mask, f"{prefix}_diag_delta")
        )
        if teacher_lp is not None:
            metrics.update(
                _sd_distribution_metrics(
                    teacher_lp, response_mask, f"{prefix}_diag_teacher_lp"
                )
            )
        if gate is not None:
            gated = mask & gate.bool()
            ungated = mask & ~gate.bool()
            metrics.update({
                f"{prefix}_diag_gated_delta_abs_mean": _sd_masked_mean(
                    abs_delta, gated
                ),
                f"{prefix}_diag_ungated_delta_abs_mean": _sd_masked_mean(
                    abs_delta, ungated
                ),
                f"{prefix}_diag_gated_clip_frac": _sd_masked_frac(
                    saturates, gated
                ),
                f"{prefix}_diag_ungated_clip_frac": _sd_masked_frac(
                    saturates, ungated
                ),
                f"{prefix}_diag_gated_teacher_lp_mean": _sd_masked_mean(
                    teacher_lp, gated
                ) if teacher_lp is not None else float("nan"),
                f"{prefix}_diag_ungated_teacher_lp_mean": _sd_masked_mean(
                    teacher_lp, ungated
                ) if teacher_lp is not None else float("nan"),
            })
            # Enrichment > 1 means the gate prefers tokens the teacher cannot
            # predict, which is the opposite of the gate's design intent.
            #
            # Emitted unconditionally, NaN when undefined: metric keys are
            # merged across ranks with list_of_dict_to_dict_of_list, so a key
            # that is present in only some micro-batches produces ragged
            # per-rank lists and crashes reduce_metrics. NaN is dropped later
            # by the ignore-NaN rule for *_diag_* keys.
            gated_abs = metrics[f"{prefix}_diag_gated_delta_abs_mean"]
            overall_abs = metrics[f"{prefix}_diag_delta_abs_mean"]
            enrichment = float("nan")
            if (
                overall_abs == overall_abs  # not NaN
                and overall_abs > 0
                and gated_abs == gated_abs
            ):
                enrichment = gated_abs / overall_abs
            metrics[f"{prefix}_diag_gate_blindness_enrichment"] = enrichment
        if gamma_t is not None:
            valid = mask & torch.isfinite(gamma_t) & torch.isfinite(delta_t)
            metrics[f"{prefix}_diag_gamma_gt20_frac"] = _sd_masked_frac(
                gamma_t > 20.0, mask
            )
            # Same rule as above: always emit, NaN when there is nothing to
            # measure, so every micro-batch contributes the same key set.
            corr = float("nan")
            argmax_gamma = float("nan")
            argmax_delta = float("nan")
            if valid.sum() > 1:
                gamma_flat = torch.masked_select(gamma_t.float(), valid)
                delta_flat = torch.masked_select(delta_t.float(), valid)
                gamma_centered = gamma_flat - gamma_flat.mean()
                delta_centered = delta_flat - delta_flat.mean()
                denom = (
                    gamma_centered.norm() * delta_centered.norm()
                ).clamp(min=1e-12)
                corr = (torch.dot(gamma_centered, delta_centered) / denom).item()
                # Locate the pathological gamma tail: report the delta at the
                # single largest-gamma token.
                arg = int(torch.argmax(gamma_flat).item())
                argmax_gamma = gamma_flat[arg].item()
                argmax_delta = delta_flat[arg].item()
            metrics[f"{prefix}_diag_gamma_delta_corr"] = corr
            metrics[f"{prefix}_diag_gamma_argmax_gamma"] = argmax_gamma
            metrics[f"{prefix}_diag_gamma_argmax_delta"] = argmax_delta
    return metrics


def _eg_rlsd_gate_contingency(delta_t, student_gate, teacher_gate, eligible_mask):
    """2x2 split of eligible tokens by the student-side and teacher-side gates.

    ``both`` is the informed-disagreement cell: the gold evidence matters to the
    student *and* the teacher's ability at that token comes from the evidence it
    holds, so the teacher ratio is trustworthy there. ``student_only`` is the
    cell where the teacher has an opinion that is not evidence-grounded, i.e.
    exactly the tokens the gate was meant to exclude. Comparing mean|delta|
    across the two cells says which one drives the modulation.
    """
    with torch.no_grad():
        eligible = eligible_mask.bool()
        student = student_gate.bool()
        teacher = teacher_gate.bool()
        cells = {
            "both": eligible & student & teacher,
            "student_only": eligible & student & ~teacher,
            "teacher_only": eligible & ~student & teacher,
            "neither": eligible & ~student & ~teacher,
        }
        abs_delta = delta_t.abs()
        metrics = {}
        for name, cell in cells.items():
            metrics[f"eg_rlsd_diag_cell_{name}_ratio"] = _sd_masked_frac(
                cell, eligible
            )
            metrics[f"eg_rlsd_diag_cell_{name}_delta_abs_mean"] = _sd_masked_mean(
                abs_delta, cell
            )
    return metrics


def _shuffle_delta_within_gate(effective_delta, gate):
    """Randomly permute delta AMONG the gated positions of each row.

    The placebo control for "does the teacher's per-token opinion carry
    information?" (2026-08-14). Every statistic the modulation depends on is
    preserved per row -- which positions are gated, how many, and the exact
    multiset of delta values, hence the same teacher_weight distribution, the
    same epsilon_w clamping behaviour and the same mean |advantage| shift. The
    ONLY thing destroyed is the pairing between a token and the teacher's
    judgment of that token. So:

        placebo == real  ->  the SD term acts as advantage noise; the teacher's
                             judgment is not what produces the effect
        placebo <  real  ->  the per-token judgment carries information

    Non-gated positions are returned as 0, exactly as the gate would leave them.
    """
    if gate is None:
        mask = torch.ones_like(effective_delta, dtype=torch.bool)
    else:
        mask = gate.to(dtype=torch.bool)
    length = effective_delta.shape[-1]
    inf = torch.full_like(effective_delta, float("inf"))
    # slots 0..c-1 of `perm` are the gated positions in RANDOM order, and the
    # same slots of `pos` are the gated positions in NATURAL order, so gathering
    # by one and scattering by the other permutes values among gated positions
    perm = torch.where(mask, torch.rand_like(effective_delta), inf).argsort(dim=-1)
    natural = torch.arange(length, device=effective_delta.device,
                           dtype=effective_delta.dtype).expand_as(effective_delta)
    pos = torch.where(mask, natural, inf).argsort(dim=-1)
    src = torch.gather(effective_delta, -1, perm)
    # beyond c both index tensors point at non-gated positions; zero the source
    # there so those positions stay 0 instead of receiving a gated value
    valid = (torch.arange(length, device=effective_delta.device)
             .expand_as(effective_delta) < mask.sum(dim=-1, keepdim=True))
    out = torch.zeros_like(effective_delta)
    out.scatter_(-1, pos, src * valid.to(dtype=effective_delta.dtype))
    return out


def _rlsd_modulated_advantages(advantages, delta_t, epsilon_w, lam, gate=None,
                               onesided_clamp=False, penalty_only=False,
                               boost_gate=None, shuffle_placebo=False):
    """Apply the shared RLSD teacher-ratio modulation, optionally behind a gate.

    ``shuffle_placebo`` permutes delta among the gated positions of each row
    before the weight is computed; see ``_shuffle_delta_within_gate``. It exists
    only to run the control experiment and must never be set on a real arm.

    ``onesided_clamp`` enforces ``w >= 1`` wherever the advantage is positive:
    token-level teacher opinion may add credit on verifier-approved
    trajectories but may not subtract it. Negative-advantage tokens are
    unaffected.

    ``penalty_only`` (2026-08-11, "change #3"): clamp the signal to its
    NEGATIVE half before the sign multiply. The teacher then never amplifies
    reward -- signal>0 tokens are untouched in every quadrant (w=1), while
    signal<0 tokens get their reward attenuated (A>0) or their punishment
    amplified (A<0). This kills the recitation ratchet (copy tokens ride the
    positive half) and makes the modulation structurally anti-sharpening,
    mirroring the property that keeps rlsd/eg entropy-stable (their delta_t
    is near-nonpositive everywhere). Offline replay on the 0809 data: copy
    -token net incentive +0.106 -> -0.017, grounded (c>0) tokens exactly
    untouched, 27.6% of tokens still modulated. Mutually exclusive with
    onesided_clamp in spirit (onesided_clamp would re-zero the A>0 tax).
    """
    with torch.no_grad():
        effective_delta = delta_t.detach()
        if gate is not None:
            if gate.shape != effective_delta.shape:
                raise AssertionError("RLSD gate shape must match token-level log probabilities")
            effective_delta = torch.where(
                gate.to(dtype=torch.bool),
                effective_delta,
                torch.zeros_like(effective_delta),
            )
        if penalty_only:
            effective_delta = effective_delta.clamp(max=0.0)
        if boost_gate is not None:
            # ``boost_gate`` (2026-08-16, "change #4"): the POSITIVE half of the
            # signal only passes where the gate is True; the negative half is
            # untouched (refutation applies even to confident fabrications).
            # Used by ce_rlsd with rlsd_boost_prob_ceiling: gate = tokens whose
            # behaviour-policy probability exp(old_log_prob) is still below the
            # ceiling. Copy/recitation tokens ride at p~=1 (0816 offline audit:
            # copy-span H p90=0.06, p_chosen~=1.00) so verbatim transcription
            # earns no boost no matter how gold-supported it is, while genuine
            # fork tokens (model uncertain + gold knows) keep the full reward.
            if boost_gate.shape != effective_delta.shape:
                raise AssertionError(
                    "RLSD boost gate shape must match token-level log probabilities"
                )
            effective_delta = torch.where(
                (effective_delta > 0) & ~boost_gate.to(dtype=torch.bool),
                torch.zeros_like(effective_delta),
                effective_delta,
            )
        scaled_delta = (advantages.sign() * effective_delta).clamp(min=-60.0, max=60.0)
        if shuffle_placebo:
            # Permute AFTER the sign multiply, not before. w = exp(sign(A)*delta)
            # and sign(A) varies within a row, so shuffling delta would re-pair
            # magnitudes with different signs and change the multiset of w --
            # i.e. it would alter the perturbation's SIZE, which is exactly the
            # confound this control exists to avoid. Shuffling scaled_delta
            # leaves the per-row multiset of w (and hence of |advantage shift|)
            # bit-identical and destroys only the token pairing.
            scaled_delta = _shuffle_delta_within_gate(scaled_delta, gate)
        teacher_weight = scaled_delta.exp().clamp(1 - epsilon_w, 1 + epsilon_w)
        if onesided_clamp:
            teacher_weight = torch.where(
                advantages > 0,
                teacher_weight.clamp(min=1.0),
                teacher_weight,
            )
        mod_advantages = advantages * (1.0 + lam * (teacher_weight - 1.0))
    return mod_advantages, teacher_weight


@register_policy_loss("rlsd")
def compute_policy_loss_rlsd(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token-mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """RLSD: vanilla GRPO/PPO with teacher-ratio modulated advantages.

    When ``rlsd_skip_math=True`` the batch carries an ``evidence_sample_mask``
    (True = non-math row). Ineligible rows have their delta zeroed inside
    ``_rlsd_modulated_advantages`` (w=1), i.e. they fall back to exact GRPO,
    mirroring the eg_rlsd/ce_rlsd eligibility contract.
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "RLSD requires teacher_log_probs but got None. "
        "teacher_mode=snapshot requires EMA/snapshot implementation in engine_workers."
    )
    assert teacher_lp.shape == log_prob.shape
    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)

    skip_math = _sd_bool_cfg(config, "rlsd_skip_math", False)
    evidence_sample_mask = kwargs.get("evidence_sample_mask")
    gate = None
    if skip_math:
        assert evidence_sample_mask is not None, (
            "rlsd_skip_math=True requires evidence_sample_mask in the batch; "
            "check _compute_teacher_log_probs attached it for loss_mode=rlsd."
        )
        if evidence_sample_mask.dim() == 1:
            evidence_sample_mask = evidence_sample_mask.unsqueeze(-1)
        if evidence_sample_mask.shape[0] != log_prob.shape[0]:
            raise AssertionError(
                "evidence_sample_mask batch dimension does not match policy log probabilities"
            )
        gate = (
            evidence_sample_mask.to(device=log_prob.device, dtype=torch.bool).expand_as(response_mask)
            & response_mask.to(dtype=torch.bool)
        )

    delta_t = teacher_lp.detach() - log_prob.detach()
    mod_advantages, teacher_weight = _rlsd_modulated_advantages(
        advantages,
        delta_t,
        epsilon_w,
        lam,
        gate=gate,
    )
    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=mod_advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    metrics.update({
        "rlsd_lam": lam,
        "rlsd_weight_abs_shift": _sd_token_mean(
            (teacher_weight - 1.0).abs(), response_mask
        ).item(),
    })
    if skip_math:
        metrics["rlsd_eligible_token_ratio"] = _sd_token_mean(
            gate.to(dtype=log_prob.dtype), response_mask.to(dtype=log_prob.dtype)
        ).item()
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    if _sd_bool_cfg(config, "evidence_diagnostics", True):
        metrics.update(
            _rlsd_diagnostics(
                delta_t,
                teacher_lp.detach(),
                response_mask,
                epsilon_w,
                "rlsd",
                advantages,
            )
        )
    return loss, metrics


@register_policy_loss("eg_rlsd")
def compute_policy_loss_eg_rlsd(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
    **kwargs,
):
    """Evidence-gated RLSD with exact GRPO fallback on ineligible tokens.

    ``teacher_log_probs`` is the privileged gold-evidence view, while
    ``evidence_removed_log_probs`` is the current policy evaluated on the long
    context with gold evidence removed. Only tokens whose probability
    decreases after evidence removal receive RLSD advantage modulation.
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    evidence_removed_lp = kwargs.get("evidence_removed_log_probs")
    evidence_sample_mask = kwargs.get("evidence_sample_mask")
    assert teacher_lp is not None, "eg_rlsd requires teacher_log_probs."
    assert evidence_removed_lp is not None, "eg_rlsd requires evidence_removed_log_probs."
    assert evidence_sample_mask is not None, "eg_rlsd requires evidence_sample_mask."
    assert teacher_lp.shape == evidence_removed_lp.shape == log_prob.shape

    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    gate_tau = max(float(_sd_cfg(config, "evidence_gate_tau", 0.0)), 0.0)
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)
    teacher_gate_mode = str(_sd_cfg(config, "teacher_gate_mode", "off")).strip().lower()
    teacher_gate_tau = float(_sd_cfg(config, "teacher_gate_tau", 0.0))
    # delta_source selects what modulates the advantages:
    #   student       : delta = teacher_lp(gold+q) - policy_lp(full doc). The
    #                   two sides differ in BOTH weights and evidence; the
    #                   20260727 probe showed this signal is one-sided and
    #                   nearly dead (p95 ~ 0.12, extremes 100% negative).
    #   gamma_teacher : delta = teacher_lp(gold+q) - teacher_lp(q only), both
    #                   under the frozen shadow weights -- a pure evidence
    #                   contrast ("does the evidence the teacher holds explain
    #                   this token"). Two-sided on data (frac>0 0.38 vs 0.18).
    #                   The student/teacher gates are bypassed: the delta IS
    #                   the contrast, so gating on the same quantity again is
    #                   redundant. Ineligible (math) rows still fall back to
    #                   exact GRPO via the eligibility mask.
    delta_source = str(
        _sd_cfg(config, "rlsd_delta_source", "student")
    ).strip().lower()
    if delta_source not in ("student", "gamma_teacher"):
        raise AssertionError(
            f"rlsd_delta_source must be 'student' or 'gamma_teacher', got {delta_source!r}"
        )
    onesided_clamp = _sd_bool_cfg(config, "rlsd_onesided_clamp", False)
    teacher_noevidence_lp = kwargs.get("teacher_noevidence_log_probs")
    if teacher_gate_mode == "gate" and teacher_noevidence_lp is None:
        raise AssertionError(
            "teacher_gate_mode=gate requires teacher_noevidence_log_probs; "
            "the teacher_noevidence_* tensors are missing from the batch."
        )
    if delta_source == "gamma_teacher" and teacher_noevidence_lp is None:
        raise AssertionError(
            "rlsd_delta_source=gamma_teacher requires teacher_noevidence_log_probs; "
            "set teacher_gate_mode=diagnose (or gate) so the no-evidence "
            "teacher forward is included in the batch."
        )

    if evidence_sample_mask.dim() == 1:
        evidence_sample_mask = evidence_sample_mask.unsqueeze(-1)
    if evidence_sample_mask.shape[0] != log_prob.shape[0]:
        raise AssertionError(
            "evidence_sample_mask batch dimension does not match policy log probabilities"
        )

    with torch.no_grad():
        eligible_mask = (
            evidence_sample_mask.to(dtype=torch.bool).expand_as(response_mask)
            & response_mask.to(dtype=torch.bool)
        )
        delta_t = teacher_lp.detach() - log_prob.detach()
        gamma_t = log_prob.detach() - evidence_removed_lp.detach()
        student_gate = eligible_mask & torch.isfinite(gamma_t) & (gamma_t > gate_tau)
        # gamma_T: how much of the teacher's own predictive ability at this
        # token comes from the evidence it holds. Student-side gamma_t cannot
        # tell an informed teacher disagreement from a teacher that simply
        # cannot see what the token depends on.
        gamma_teacher = None
        teacher_gate = None
        if teacher_noevidence_lp is not None:
            if teacher_noevidence_lp.shape != log_prob.shape:
                raise AssertionError(
                    "teacher_noevidence_log_probs shape must match log_prob"
                )
            gamma_teacher = teacher_lp.detach() - teacher_noevidence_lp.detach()
            teacher_gate = torch.isfinite(gamma_teacher) & (
                gamma_teacher > teacher_gate_tau
            )
        if delta_source == "gamma_teacher":
            # Pure evidence contrast under one set of (frozen) weights. Only
            # the eligibility mask remains: math rows carry no evidence views
            # and must stay exact GRPO.
            delta_used = gamma_teacher
            gate = eligible_mask & torch.isfinite(gamma_teacher)
        else:
            delta_used = delta_t
            if teacher_gate_mode == "gate":
                gate = student_gate & teacher_gate
            else:
                gate = student_gate
        mod_advantages, teacher_weight = _rlsd_modulated_advantages(
            advantages,
            delta_used,
            epsilon_w,
            lam,
            gate=gate,
            onesided_clamp=onesided_clamp,
        )

    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=mod_advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    response_mask_float = response_mask.to(dtype=log_prob.dtype)
    eligible_mask_float = eligible_mask.to(dtype=log_prob.dtype)
    gate_float = gate.to(dtype=log_prob.dtype)
    eligible_count = eligible_mask_float.sum()
    metrics.update({
        "eg_rlsd_lam": lam,
        "eg_rlsd_gate_tau": gate_tau,
        "eg_rlsd_eligible_token_ratio": _sd_token_mean(
            eligible_mask_float, response_mask_float
        ).item(),
        "eg_rlsd_gate_ratio": _sd_token_mean(
            gate_float, response_mask_float
        ).item(),
        "eg_rlsd_gate_ratio_on_eligible": (
            gate_float.sum() / eligible_count.clamp(min=1)
        ).item(),
        "eg_rlsd_weight_abs_shift": _sd_token_mean(
            (teacher_weight - 1.0).abs(), response_mask_float
        ).item(),
    })
    teacher_metrics = _sd_teacher_metrics(
        teacher_lp,
        log_prob,
        eligible_mask_float,
        kwargs.get("teacher_entropy"),
    )
    if eligible_count.item() == 0:
        teacher_metrics = {key: float("nan") for key in teacher_metrics}
    metrics.update(teacher_metrics)
    metrics.update(
        _sd_distribution_metrics(
            gamma_t,
            eligible_mask_float,
            "eg_rlsd_gamma",
            include_percentiles=False,
        )
    )
    metrics["eg_rlsd_teacher_gate_mode"] = float(
        {"off": 0.0, "diagnose": 1.0, "gate": 2.0}.get(teacher_gate_mode, 0.0)
    )
    metrics["eg_rlsd_delta_source"] = float(
        {"student": 0.0, "gamma_teacher": 1.0}.get(delta_source, 0.0)
    )
    metrics["eg_rlsd_onesided_clamp"] = float(bool(onesided_clamp))
    # Keyed off the config, not off gamma_teacher, so every micro-batch of a
    # given run contributes an identical key set. Metrics are merged across
    # ranks positionally, so a key that appears in only some micro-batches
    # yields ragged per-rank lists and crashes reduce_metrics.
    if teacher_gate_mode != "off":
        metrics.update({
            "eg_rlsd_teacher_gate_tau": teacher_gate_tau,
            "eg_rlsd_gamma_teacher_ratio": (
                _sd_masked_frac(teacher_gate, eligible_mask)
                if teacher_gate is not None
                else float("nan")
            ),
            # Sentinel: the no-evidence view has the shortest prompt of all
            # four views, so it is the first thing a regressed per-row aligner
            # would zero out. A mean near 0.0 means zero padding, not a
            # confident teacher.
            "eg_rlsd_teacher_noevidence_avg_logprob": (
                _sd_masked_mean(teacher_noevidence_lp.detach(), eligible_mask)
                if teacher_noevidence_lp is not None
                else float("nan")
            ),
        })
    if _sd_bool_cfg(config, "evidence_diagnostics", True):
        metrics.update(
            _sd_distribution_metrics(
                gamma_t, eligible_mask_float, "eg_rlsd_diag_gamma"
            )
        )
        metrics.update(
            _rlsd_diagnostics(
                delta_used,
                teacher_lp.detach(),
                eligible_mask,
                epsilon_w,
                "eg_rlsd",
                advantages,
                gate=gate,
                gamma_t=gamma_t,
            )
        )
        if teacher_gate_mode != "off":
            metrics.update(
                _sd_distribution_metrics(
                    gamma_teacher
                    if gamma_teacher is not None
                    else torch.full_like(delta_t, float("nan")),
                    eligible_mask_float,
                    "eg_rlsd_diag_gamma_teacher",
                )
            )
            metrics.update(
                _eg_rlsd_gate_contingency(
                    delta_t,
                    student_gate,
                    teacher_gate
                    if teacher_gate is not None
                    else torch.zeros_like(student_gate),
                    eligible_mask,
                )
            )
    return loss, metrics


@register_policy_loss("ce_gate_rlsd")
def compute_policy_loss_ce_gate_rlsd(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
    **kwargs,
):
    """RLSD gated by the magnitude of a teacher-side evidence contrast.

    The gate and value deliberately answer different questions:

      ce_t = log p_T(y_t | gold + question) - log p_T(y_t | question)
      gate_t = eligible_t and |ce_t| >= ce_gate_tau
      value_t = log p_T(y_t | gold + question) - log p_S(y_t | full context)

    Both signs of CE are informative: positive CE means the evidence supports
    the sampled token, while negative CE means it refutes that token. Tokens
    outside the gate have teacher weight exactly one and therefore reduce to
    vanilla GRPO.
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    teacher_noevidence_lp = kwargs.get("teacher_noevidence_log_probs")
    evidence_sample_mask = kwargs.get("evidence_sample_mask")
    assert teacher_lp is not None, "ce_gate_rlsd requires teacher_log_probs."
    assert teacher_noevidence_lp is not None, (
        "ce_gate_rlsd requires teacher_noevidence_log_probs; the "
        "teacher_noevidence_* tensors are missing from the batch."
    )
    assert evidence_sample_mask is not None, (
        "ce_gate_rlsd requires evidence_sample_mask."
    )
    assert teacher_lp.shape == teacher_noevidence_lp.shape == log_prob.shape

    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    gate_tau = max(float(_sd_cfg(config, "ce_gate_tau", 0.2)), 0.0)
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)

    if evidence_sample_mask.dim() == 1:
        evidence_sample_mask = evidence_sample_mask.unsqueeze(-1)
    if evidence_sample_mask.shape[0] != log_prob.shape[0]:
        raise AssertionError(
            "evidence_sample_mask batch dimension does not match policy log probabilities"
        )

    with torch.no_grad():
        eligible_mask = (
            evidence_sample_mask.to(dtype=torch.bool).expand_as(response_mask)
            & response_mask.to(dtype=torch.bool)
        )
        ce_t = teacher_lp.detach() - teacher_noevidence_lp.detach()
        delta_ts = teacher_lp.detach() - log_prob.detach()
        finite_mask = torch.isfinite(ce_t) & torch.isfinite(delta_ts)
        full_rlsd_gate = eligible_mask & finite_mask
        gate = full_rlsd_gate & (ce_t.abs() >= gate_tau)

        # placebo control (2026-08-14): permute delta among the gated positions.
        # Applied ONLY to the advantages that feed the loss; the diagnostic
        # full-gate call below stays unshuffled so retained_shift_mass_ratio
        # keeps comparing like with like.
        shuffle_placebo = bool(_sd_cfg(config, "rlsd_shuffle_placebo", False))
        mod_advantages, teacher_weight = _rlsd_modulated_advantages(
            advantages,
            delta_ts,
            epsilon_w,
            lam,
            gate=gate,
            shuffle_placebo=shuffle_placebo,
        )
        full_rlsd_advantages, _ = _rlsd_modulated_advantages(
            advantages,
            delta_ts,
            epsilon_w,
            lam,
            gate=full_rlsd_gate,
        )
        gated_shift_mass = (mod_advantages - advantages).abs().sum()
        full_rlsd_shift_mass = (full_rlsd_advantages - advantages).abs().sum()
        retained_shift_mass_ratio = (
            (gated_shift_mass / full_rlsd_shift_mass).item()
            if full_rlsd_shift_mass.item() > 0.0
            else float("nan")
        )

    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=mod_advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    response_mask_float = response_mask.to(dtype=log_prob.dtype)
    eligible_mask_float = eligible_mask.to(dtype=log_prob.dtype)
    gate_float = gate.to(dtype=log_prob.dtype)
    eligible_count = eligible_mask_float.sum()
    positive_ce_mask = full_rlsd_gate & (ce_t > 0)
    negative_ce_mask = full_rlsd_gate & (ce_t < 0)
    metrics.update({
        "ce_gate_rlsd_lam": lam,
        "ce_gate_rlsd_gate_tau": gate_tau,
        # self-documenting: a placebo run is identifiable from its metrics alone
        "ce_gate_rlsd_placebo_shuffle": 1.0 if shuffle_placebo else 0.0,
        "ce_gate_rlsd_eligible_token_ratio": _sd_token_mean(
            eligible_mask_float, response_mask_float
        ).item(),
        "ce_gate_rlsd_gate_ratio": _sd_token_mean(
            gate_float, response_mask_float
        ).item(),
        "ce_gate_rlsd_gate_ratio_on_eligible": (
            gate_float.sum() / eligible_count.clamp(min=1)
        ).item(),
        "ce_gate_rlsd_ce_positive_retained_frac": _sd_masked_frac(
            gate, positive_ce_mask
        ),
        "ce_gate_rlsd_ce_negative_retained_frac": _sd_masked_frac(
            gate, negative_ce_mask
        ),
        "ce_gate_rlsd_weight_abs_shift": _sd_token_mean(
            (teacher_weight - 1.0).abs(), response_mask_float
        ).item(),
        "ce_gate_rlsd_advantage_abs_shift": _sd_token_mean(
            (mod_advantages - advantages).abs(), response_mask_float
        ).item(),
        "ce_gate_rlsd_retained_full_rlsd_shift_mass_ratio": retained_shift_mass_ratio,
        # A mean near 0.0 indicates zero padding from a broken response
        # aligner, not an unusually confident question-only teacher.
        "ce_gate_rlsd_teacher_noevidence_avg_logprob": _sd_masked_mean(
            teacher_noevidence_lp.detach(), eligible_mask
        ),
    })
    teacher_metrics = _sd_teacher_metrics(
        teacher_lp,
        log_prob,
        eligible_mask_float,
        kwargs.get("teacher_entropy"),
    )
    if eligible_count.item() == 0:
        teacher_metrics = {key: float("nan") for key in teacher_metrics}
    metrics.update(teacher_metrics)
    if _sd_bool_cfg(config, "evidence_diagnostics", True):
        metrics.update(
            _sd_distribution_metrics(
                ce_t,
                eligible_mask_float,
                "ce_gate_rlsd_diag_ce",
            )
        )
        metrics.update(
            _rlsd_diagnostics(
                delta_ts,
                teacher_lp.detach(),
                eligible_mask,
                epsilon_w,
                "ce_gate_rlsd",
                advantages,
                gate=gate,
            )
        )
    return loss, metrics


@register_policy_loss("ce_rlsd")
def compute_policy_loss_ce_rlsd(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
    **kwargs,
):
    """Contrastive-Evidence RLSD: pure teacher-side evidence contrast.

    delta_t = teacher_lp(gold + question) - teacher_lp(question only), both
    under the same frozen shadow weights, so the two sides of the ratio differ
    ONLY in the evidence they hold. This removes the two confounds of the
    legacy RLSD denominator (student on the full 15k document): the
    weight-drift confound and the context-length/predictability confound that
    the 20260727 probe showed makes the legacy delta one-sided and nearly dead.

    No boolean gates: filler tokens self-gate at delta ~ 0 and the contrast
    itself is the signal (gating on the same quantity again is redundant).
    Only the eligibility mask remains -- math rows carry no evidence views and
    fall back to exact GRPO.

    Degeneracy: if the two teacher views coincide (delta == 0 everywhere) the
    loss is exactly vanilla GRPO. This differs from CEPO's answer-contrast,
    whose degenerate case (P_T- == P_S) is RLSD.

    Cost: two short teacher forwards (~1.7k + ~0.1k prompt tokens); no
    evidence_removed (minusgold, ~15k) forward, unlike eg_rlsd with
    rlsd_delta_source=gamma_teacher.
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    teacher_noevidence_lp = kwargs.get("teacher_noevidence_log_probs")
    evidence_sample_mask = kwargs.get("evidence_sample_mask")
    assert teacher_lp is not None, "ce_rlsd requires teacher_log_probs."
    assert teacher_noevidence_lp is not None, (
        "ce_rlsd requires teacher_noevidence_log_probs; the "
        "teacher_noevidence_* tensors are missing from the batch."
    )
    assert evidence_sample_mask is not None, "ce_rlsd requires evidence_sample_mask."
    assert teacher_lp.shape == teacher_noevidence_lp.shape == log_prob.shape

    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)
    onesided_clamp = _sd_bool_cfg(config, "rlsd_onesided_clamp", False)
    penalty_only = _sd_bool_cfg(config, "rlsd_penalty_only", False)
    shuffle_placebo = _sd_bool_cfg(config, "rlsd_shuffle_placebo", False)
    if penalty_only and onesided_clamp:
        raise ValueError(
            "rlsd_penalty_only and rlsd_onesided_clamp are mutually exclusive: "
            "onesided_clamp re-zeroes the A>0 tax that penalty_only relies on."
        )
    boost_prob_ceiling = _sd_cfg(config, "rlsd_boost_prob_ceiling", None)
    if boost_prob_ceiling is not None:
        boost_prob_ceiling = float(boost_prob_ceiling)
        if not 0.0 < boost_prob_ceiling <= 1.0:
            raise ValueError(
                f"rlsd_boost_prob_ceiling must be in (0, 1], got {boost_prob_ceiling}"
            )
        if penalty_only:
            raise ValueError(
                "rlsd_boost_prob_ceiling and rlsd_penalty_only are mutually "
                "exclusive: penalty_only zeroes the whole positive half, leaving "
                "the boost gate nothing to gate."
            )
        if onesided_clamp:
            raise ValueError(
                "rlsd_boost_prob_ceiling and rlsd_onesided_clamp are mutually "
                "exclusive: the clamp re-opens the A>0 reward the gate blocks."
            )

    if evidence_sample_mask.dim() == 1:
        evidence_sample_mask = evidence_sample_mask.unsqueeze(-1)
    if evidence_sample_mask.shape[0] != log_prob.shape[0]:
        raise AssertionError(
            "evidence_sample_mask batch dimension does not match policy log probabilities"
        )

    with torch.no_grad():
        eligible_mask = (
            evidence_sample_mask.to(dtype=torch.bool).expand_as(response_mask)
            & response_mask.to(dtype=torch.bool)
        )
        delta_t = teacher_lp.detach() - teacher_noevidence_lp.detach()
        gate = eligible_mask & torch.isfinite(delta_t)
        boost_gate = None
        if boost_prob_ceiling is not None:
            boost_gate = old_log_prob.detach().exp() < boost_prob_ceiling
        mod_advantages, teacher_weight = _rlsd_modulated_advantages(
            advantages,
            delta_t,
            epsilon_w,
            lam,
            gate=gate,
            onesided_clamp=onesided_clamp,
            penalty_only=penalty_only,
            boost_gate=boost_gate,
            shuffle_placebo=shuffle_placebo,
        )

    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=mod_advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # Metric keys must be identical for every micro-batch of a run: metrics
    # are merged across ranks positionally, and a conditionally-present key
    # produces ragged per-rank lists that crash reduce_metrics. Values default
    # to NaN (covered by the ignore-NaN rules) when there is nothing to
    # measure.
    response_mask_float = response_mask.to(dtype=log_prob.dtype)
    eligible_mask_float = eligible_mask.to(dtype=log_prob.dtype)
    eligible_count = eligible_mask_float.sum()
    negative_ce_mask = gate & (delta_t < 0)
    positive_ce_mask = gate & (delta_t > 0)
    if boost_prob_ceiling is not None:
        boosted_mask = positive_ce_mask & boost_gate
    else:
        boosted_mask = positive_ce_mask
    positive_ce_count = positive_ce_mask.to(dtype=log_prob.dtype).sum()
    effective_penalty_mask = gate & ((mod_advantages - advantages).abs() > 0)
    advantage_abs_mass = (advantages.abs() * response_mask_float).sum()
    advantage_shift_abs_mass = (
        (mod_advantages - advantages).abs() * response_mask_float
    ).sum()
    metrics.update({
        "ce_rlsd_lam": lam,
        "ce_rlsd_eligible_token_ratio": _sd_token_mean(
            eligible_mask_float, response_mask_float
        ).item(),
        "ce_rlsd_modulated_token_ratio": _sd_token_mean(
            gate.to(dtype=log_prob.dtype), response_mask_float
        ).item(),
        "ce_rlsd_weight_abs_shift": _sd_token_mean(
            (teacher_weight - 1.0).abs(), response_mask_float
        ).item(),
        "ce_rlsd_advantage_abs_shift": _sd_token_mean(
            (mod_advantages - advantages).abs(), response_mask_float
        ).item(),
        "ce_rlsd_advantage_rel_abs_shift": (
            (advantage_shift_abs_mass / advantage_abs_mass).item()
            if advantage_abs_mass.item() > 0.0
            else float("nan")
        ),
        "ce_rlsd_negative_ce_token_ratio": _sd_token_mean(
            negative_ce_mask.to(dtype=log_prob.dtype), response_mask_float
        ).item(),
        "ce_rlsd_effective_penalty_token_ratio": _sd_token_mean(
            effective_penalty_mask.to(dtype=log_prob.dtype), response_mask_float
        ).item(),
        "ce_rlsd_onesided_clamp": float(bool(onesided_clamp)),
        "ce_rlsd_penalty_only": float(bool(penalty_only)),
        # Boost gate (change #4). Keys are always present (constant key set per
        # run); ceiling is NaN when the gate is off. gate_ratio = share of
        # positive-signal tokens that pass, expected ~0.3 at step 1 and
        # MONOTONICALLY DECLINING (self-limiting: learned tokens exit the band).
        "ce_rlsd_boost_prob_ceiling": (
            float(boost_prob_ceiling)
            if boost_prob_ceiling is not None
            else float("nan")
        ),
        "ce_rlsd_boost_gate_ratio": (
            (boosted_mask.to(dtype=log_prob.dtype).sum() / positive_ce_count).item()
            if positive_ce_count.item() > 0.0
            else float("nan")
        ),
        "ce_rlsd_boost_token_frac": _sd_token_mean(
            boosted_mask.to(dtype=log_prob.dtype), response_mask_float
        ).item(),
        # Runtime proof that the config flag reached this loss. A missing
        # allowlist/schema hop otherwise produces a clean-looking duplicate of
        # the real arm.
        "ce_rlsd_placebo_shuffle": float(bool(shuffle_placebo)),
        # Sentinel: the question-only view has the shortest prompt of all
        # views, so it is the first tensor a regressed per-row aligner would
        # zero out. A mean near 0.0 means zero padding, not a confident
        # teacher.
        "ce_rlsd_teacher_noevidence_avg_logprob": _sd_masked_mean(
            teacher_noevidence_lp.detach(), eligible_mask
        ),
    })
    teacher_metrics = _sd_teacher_metrics(
        teacher_lp,
        log_prob,
        eligible_mask_float,
        kwargs.get("teacher_entropy"),
    )
    if eligible_count.item() == 0:
        teacher_metrics = {key: float("nan") for key in teacher_metrics}
    metrics.update(teacher_metrics)
    if _sd_bool_cfg(config, "evidence_diagnostics", True):
        metrics.update(
            _rlsd_diagnostics(
                delta_t,
                teacher_lp.detach(),
                eligible_mask,
                epsilon_w,
                "ce_rlsd",
                advantages,
            )
        )
    return loss, metrics


@register_policy_loss("grpo_opsd_ectr")
def compute_policy_loss_grpo_opsd_ectr(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
    **kwargs,
):
    """GRPO policy gradient PLUS the OPSD contrastive-guided distill term.

        loss = pg_loss(advantages) + distill_coef * JSD(softmax(guided) || student)

    The teacher contrast enters ONLY through the distill target:

        guided = (1+alpha)*log p_T(evidence) - alpha*log p_T(no-evidence)

    Advantages are NOT touched. That is the whole difference from
    ``ce_rlsd_opsd_ectr``, which additionally modulates them by the same
    contrast, and the reason this mode exists as its own entry rather than as
    ``ce_rlsd_opsd_ectr`` with ``lam=0``:

      - lam=0 IS bit-exact on the advantage path (the modulation's only exit is
        ``A * (1.0 + lam*(w-1.0))``, and every other RLSD knob funnels through
        it), but the identity is an arithmetic cancellation that depends on one
        config field which every ce_rlsd sibling sets to 0.5. Inheriting a
        config and forgetting that field silently turns the arm into the
        hybrid, and nothing in the metrics screams.
      - Here the identity is structural: this function never calls
        ``_rlsd_modulated_advantages``, so no config can reintroduce the
        perturbation.

    The reward-shaped part is byte-identical to ``vanilla``, so distill_coef=0
    reproduces GRPO exactly. Both landmines from ``grpo_opsd`` carry over
    unchanged: ``jsd_token_clip`` must be 0, and ``distill_coef`` is not a
    fraction of the update (calibrate on ``actor/distill_over_pg``).

    Cost note: the guided target needs both teacher views, so this pays
    ce_rlsd's two teacher forwards even though it spends nothing on the
    advantage side. ``opsd_ectr_alpha=0`` drops the contrast from the target and
    makes it plain privileged-view distillation -- at which point ``grpo_opsd``
    is the cheaper mode (one teacher forward).
    """
    distill_coef = float(_sd_cfg(config, "distill_coef", 0.0))
    pg_loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights)
    metrics["pg_loss_component"] = pg_loss.detach().item()
    metrics["distill_coef"] = distill_coef

    if distill_coef == 0.0:
        # True GRPO control even if the teacher plumbing is misconfigured; see
        # _opsd_ectr_metric_keys for why the keys still have to be emitted.
        for key in _opsd_ectr_metric_keys(config):
            metrics[key] = float("nan")
        metrics["distill_weighted"] = 0.0
        metrics["distill_over_pg"] = 0.0
        return pg_loss, metrics

    assert kwargs.get("teacher_wrong_topk_log_probs") is not None, (
        "grpo_opsd_ectr needs the no-evidence teacher's log-probs gathered at the "
        "evidence teacher's top-k indices (dp_actor passes the teacher_noevidence "
        "view as teacher_wrong_topk_log_probs). Set full_logit_distill=true and "
        "top_k_distill>0, or distill_coef=0 to run pure GRPO."
    )
    distill_loss, distill_metrics = _opsd_ectr_distill_term(
        log_prob, response_mask, config, rollout_is_weights, **kwargs)
    metrics.update(distill_metrics)
    weighted = distill_coef * distill_loss
    metrics["distill_weighted"] = weighted.detach().item()
    metrics["distill_over_pg"] = abs(metrics["distill_weighted"]) / (abs(metrics["pg_loss_component"]) + 1e-8)
    return pg_loss + weighted, metrics


@register_policy_loss("ce_rlsd_opsd_ectr")
def compute_policy_loss_ce_rlsd_opsd_ectr(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
    **kwargs,
):
    """CE-RLSD policy gradient PLUS the OPSD contrastive-guided distill term.

        loss = pg_loss(A^CE) + distill_coef * JSD(softmax(guided) || student)

    Both halves read the SAME two teacher views, so this costs no forward beyond
    what ce_rlsd already pays -- only the top-k gathers:

        advantage side (ce_rlsd):  d_ce  = T(evidence) - T(no-evidence), per
                                   sampled token, modulating A multiplicatively
        distill side (opsd_ectr):  guided = (1+alpha)*log p_T(evidence)
                                            - alpha*log p_T(no-evidence)
                                   over the evidence view's top-k support

    With the ptr teacher (teacher_view=ptr_contrastive) that is
    T(full doc + pointer -> gold) against T(full doc + pointer -> random), i.e.
    the same contrast the CE weight uses, but carried as a dense target over the
    top-k vocab instead of a scalar per sampled token.

    Relationship to the neighbours:
      - ``distill_coef=0`` skips the distill graph entirely and this is
        BYTE-IDENTICAL to ``ce_rlsd`` -- the free control, same contract as
        ``grpo_opsd`` at coef 0.
      - ``opsd_ectr_alpha=0`` collapses the guided target to the evidence-view
        teacher alone, i.e. plain OPSD distillation onto the privileged view,
        while the CE contrast still modulates the advantage.
      - ``loss_mode=opsd_ectr`` is this term WITHOUT the policy gradient.

    Two carried-over landmines, both measured, not assumed:
      - ``jsd_token_clip`` MUST be 0 here. The one-sided per-vocab clamp breaks
        Gibbs' inequality (the minimizer is an entropy-inflated student, not the
        teacher) and is the diagnosed cause of both prior OPSD collapses.
      - ``distill_coef`` is NOT a fraction of the update: the raw distill
        gradient ran ~30x the PG gradient at clip 0 on Qwen3-1.7B, so calibrate
        against ``actor/distill_over_pg`` before spending a batch on a ladder.
    """
    distill_coef = float(_sd_cfg(config, "distill_coef", 0.0))

    pg_loss, metrics = compute_policy_loss_ce_rlsd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        **kwargs,
    )
    metrics["pg_loss_component"] = pg_loss.detach().item()
    metrics["distill_coef"] = distill_coef

    if distill_coef == 0.0:
        # True ce_rlsd control even if the distill plumbing is misconfigured:
        # skip the graph rather than multiplying it by zero. NaN marks
        # "not measured" (see _opsd_ectr_metric_keys for why the keys still
        # have to be here).
        for key in _opsd_ectr_metric_keys(config):
            metrics[key] = float("nan")
        metrics["distill_weighted"] = 0.0
        metrics["distill_over_pg"] = 0.0
        return pg_loss, metrics

    assert kwargs.get("teacher_wrong_topk_log_probs") is not None, (
        "ce_rlsd_opsd_ectr needs the no-evidence teacher's log-probs gathered at "
        "the evidence teacher's top-k indices (dp_actor passes the "
        "teacher_noevidence view as teacher_wrong_topk_log_probs). Set "
        "full_logit_distill=true and top_k_distill>0, or distill_coef=0 to run "
        "pure ce_rlsd."
    )
    distill_loss, distill_metrics = _opsd_ectr_distill_term(
        log_prob, response_mask, config, rollout_is_weights, **kwargs)
    metrics.update(distill_metrics)
    weighted = distill_coef * distill_loss
    metrics["distill_weighted"] = weighted.detach().item()
    metrics["distill_over_pg"] = abs(metrics["distill_weighted"]) / (abs(metrics["pg_loss_component"]) + 1e-8)
    return pg_loss + weighted, metrics


@register_policy_loss("rlsd_ectr")
def compute_policy_loss_rlsd_ectr(old_log_prob, log_prob, advantages, response_mask,
                                  loss_agg_mode="token_mean", config=None,
                                  rollout_is_weights=None, **kwargs):
    """RLSD variant: contrast teacher_correct vs teacher_wrong instead of student.

    Structurally identical to RLSD but the evidence ratio uses the pairwise
    teacher contrast e_ctr = teacher_correct_lp - teacher_wrong_lp in place of
    e_c = teacher_correct_lp - student_lp. Both correct and wrong teacher
    contexts come from the single-pair data path.
    Added a -60/60 clamp on the exponent because e_ctr (teacher vs teacher) can
    be much larger in magnitude than e_c (teacher vs student); epsilon_w still
    bounds the final weight.
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    teacher_wrong_lp = kwargs.get("teacher_wrong_log_probs")
    assert teacher_correct_lp is not None and teacher_wrong_lp is not None, (
        "rlsd_ectr requires teacher_log_probs and teacher_wrong_log_probs."
    )
    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()

    with torch.no_grad():
        delta_t = teacher_correct_lp - teacher_wrong_lp
        scaled = (advantages.sign() * delta_t).clamp(min=-60.0, max=60.0)
        w_t = scaled.exp().clamp(1 - epsilon_w, 1 + epsilon_w)
        mod_adv = advantages * ((1 - lam) + lam * w_t)

    surr1 = ratio * mod_adv
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * mod_adv
    per_token_loss = -torch.min(surr1, surr2)
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {
        "pg_loss": loss.item(),
        "pg_clipfrac": ((ratio - 1).abs() > epsilon).float().mean().item(),
        "rlsd_ectr_lam": lam,
        "rlsd_ectr_e_ctr_abs_mean": verl_F.masked_mean(delta_t.abs(), response_mask).item()}
    metrics.update(_sd_distribution_metrics(delta_t, response_mask, "rlsd_ectr_e_ctr", include_percentiles=False))
    metrics.update(_sd_teacher_metrics(teacher_correct_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("rlcsd")
def compute_policy_loss_rlcsd(old_log_prob, log_prob, advantages, response_mask,
                                loss_agg_mode="token_mean", config=None,
                                rollout_is_weights=None, **kwargs):
    """RLCSD: K-marginal wrong-hint contrast policy loss.

    Wrong-hint term is replaced by the
    log of the uniform mixture of teacher distributions over up to K non-self
    negative siblings:

        e_ctr_t = log P_T(y_t | x, y_c*, y_<t)
                  - log[ (1/K) * sum_k P_T(y_t | x, y_w^(k), y_<t) ]

    Implemented as logsumexp(teacher_wrong_multi_lp) - log(K_valid) per row,
    with -inf masking on padded K-slots so they don't contribute. All other
    hyperparameters reuse the rlcsd_* keys (tau/beta/lam/delta/eta/clip)
    so 
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    teacher_wrong_multi_lp = kwargs.get("teacher_wrong_multi_log_probs")
    teacher_wrong_multi_valid_mask = kwargs.get("teacher_wrong_multi_valid_mask")
    assert teacher_correct_lp is not None, "rlcsd_5 requires teacher_log_probs."
    assert teacher_wrong_multi_lp is not None and teacher_wrong_multi_valid_mask is not None, (
        "rlcsd_5 requires teacher_wrong_multi_log_probs and teacher_wrong_multi_valid_mask."
    )

    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    tau = max(float(_sd_cfg(config, "rlcsd_tau", _sd_cfg(config, "tau", 0.03))), 1e-6)
    beta = float(_sd_cfg(config, "rlcsd_beta", 0.2))
    lam = float(_sd_cfg(config, "rlcsd_lam", _sd_cfg(config, "lam", 0.5)))
    delta = max(float(_sd_cfg(config, "rlcsd_delta", 0.02)), 0.0)
    eta = max(float(_sd_cfg(config, "rlcsd_eta", 1.0)), 0.0)
    residual_clip_low = float(_sd_cfg(config, "rlcsd_residual_clip_low", -0.5))
    residual_clip_high = float(_sd_cfg(config, "rlcsd_residual_clip_high", 0.5))
    if residual_clip_low > residual_clip_high:
        residual_clip_low, residual_clip_high = residual_clip_high, residual_clip_low

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

    with torch.no_grad():
        # Marginalize wrong-hint logp over K valid siblings:
        # log_marginal_w = logsumexp_k(lp_k) - log(K_valid)
        valid_mask_bool = teacher_wrong_multi_valid_mask.to(dtype=torch.bool)
        masked_w_lp = teacher_wrong_multi_lp.masked_fill(
            ~valid_mask_bool.unsqueeze(-1), float("-inf"))
        K_valid = valid_mask_bool.to(dtype=teacher_wrong_multi_lp.dtype).sum(dim=1).clamp(min=1.0)
        log_marginal_w = torch.logsumexp(masked_w_lp, dim=1) - torch.log(K_valid).unsqueeze(-1)

        e_ctr = teacher_correct_lp - log_marginal_w
        s_t = torch.tanh(e_ctr / tau)
        preclip_residual = beta * lam * s_t
        residual = preclip_residual.clamp(min=residual_clip_low, max=residual_clip_high)
        selected = (residual.abs() > delta) & response_mask.bool()
        selected_float = selected.float()
        selected_count = selected_float.sum()
        residual_for_adv = selected_float * residual
        response_token_count = response_mask.sum()
        if selected_float.dim() >= 2:
            valid_sample_mask = response_mask.sum(dim=-1) > 0
            sample_count = valid_sample_mask.float().sum()
        else:
            sample_count = torch.as_tensor(1.0, dtype=selected_count.dtype, device=selected_count.device)
        raw_mod_adv = advantages + residual_for_adv
        mod_adv = torch.where(
            advantages > 0,
            raw_mod_adv.clamp_min(0.0),
            torch.where(advantages < 0, raw_mod_adv.clamp_max(0.0), torch.zeros_like(raw_mod_adv)))
        K_valid_mean = K_valid.mean()

    ratio_clamped = ratio.clamp(1 - epsilon, 1 + epsilon)
    nonselected_mask = response_mask * (1.0 - selected_float)
    selected_mask = response_mask * selected_float

    surr_nonselected1 = ratio * advantages
    surr_nonselected2 = ratio_clamped * advantages
    nonselected_per_token_loss = -torch.min(surr_nonselected1, surr_nonselected2)

    surr_selected1 = ratio * mod_adv
    surr_selected2 = ratio_clamped * mod_adv
    selected_per_token_loss = -torch.min(surr_selected1, surr_selected2)
    if rollout_is_weights is not None:
        nonselected_per_token_loss = nonselected_per_token_loss * rollout_is_weights
        selected_per_token_loss = selected_per_token_loss * rollout_is_weights
    nonselected_loss = _sd_token_mean(nonselected_per_token_loss, nonselected_mask)
    selected_loss = _sd_token_mean(selected_per_token_loss, selected_mask)
    loss = nonselected_loss + eta * selected_loss

    metrics = {
        "pg_loss": loss.item(),
        "pg_clipfrac": verl_F.masked_mean(((ratio - 1).abs() > epsilon).float(), response_mask).item(),
        "ppo_kl": ppo_kl.item(),
        "rlcsd_selected_token_count": selected_count.item(),
        "rlcsd_response_token_count": response_token_count.item(),
        "rlcsd_sample_count": sample_count.item(),
        "rlcsd_K_valid_mean": K_valid_mean.item()}
    metrics.update(_sd_distribution_metrics(e_ctr, response_mask, "rlcsd_e_ctr", include_percentiles=False))
    metrics.update(_sd_distribution_metrics(residual, selected_mask, "rlcsd_selected_residual"))
    return loss, metrics


@register_policy_loss("sdpo")
def compute_policy_loss_sdpo(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """SDPO: JSD / KL distillation with optional full-vocab or top-k views and IS correction."""
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "SDPO requires teacher_log_probs but got None. "
        "teacher_mode=ema requires EMA implementation in engine_workers."
    )
    alpha = float(_sd_cfg(config, "alpha", 0.5))
    is_clip = float(_sd_cfg(config, "is_clip", 2.0))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="student")
        per_token_kl = _sd_generalized_jsd(student_distill, teacher_distill, alpha)
    else:
        if alpha != 1.0:
            raise AssertionError("Non-full-logit SDPO only supports alpha=1.0 (reverse KL estimator).")
        log_ratio = log_prob - teacher_lp.detach()
        per_token_kl = log_ratio.detach() * log_prob

    is_ratio = (log_prob - old_log_prob).detach().clamp(-20, 20).exp().clamp(max=is_clip)
    per_token_loss = per_token_kl * is_ratio
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {
        "distill_loss": loss.item(),
        "is_ratio": is_ratio.mean().item(),
        "distill_alpha": alpha,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill)}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("srpo")
def compute_policy_loss_srpo(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """SRPO: Sample-Routed Policy Optimization (arXiv:2604.02288).

    Routes each rollout to the supervision signal best suited to its learning
    status: correct rollouts go through GRPO's PPO-clip surrogate (reward-aligned,
    sequence-level advantage), while incorrect rollouts with available teacher
    information go through SDPO's logit-level distillation, reweighted token-wise
    by teacher entropy (Dynamic-Weighted SDPO).

    The combined objective follows Eq. (Lfinal) of the paper:
        L = (Σ_{i,t} z^GRPO_i ℓ^GRPO_{i,t} + Σ_{i,t} z^SDPO_i ℓ^DW-SDPO_{i,t})
            / (Σ_{i,t} z^GRPO_i + Σ_{i,t} z^SDPO_i)
    where z^SDPO_i = (1 - c_i) m_i and z^GRPO_i = 1 - z^SDPO_i.

    The correctness flag c_i is read from the per-rollout sign of the GRPO
    advantage (a positive group-normalized advantage corresponds to a higher
    reward than the group mean, equivalent to "correct" in the binary-reward
    regime when the group is mixed). Teacher availability m_i is taken to be 1
    whenever teacher_log_probs are present.

    Hyperparameters (read from config.policy_loss):
        srpo_beta:           dynamic-weighting temperature β (default 1.0).
        alpha:               JSD mixing coefficient for the SDPO branch
                             (0 = forward KL, 1 = reverse KL, 0<α<1 = JSD).
        full_logit_distill:  use full-vocab JSD (True) or k1 reverse-KL (False).
        top_k_distill:       sparse top-k support size for the JSD branch.
        is_clip:             upper clamp on the SDPO IS ratio (default 2.0).
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    teacher_ent = kwargs.get("teacher_entropy")
    assert teacher_lp is not None, (
        "SRPO requires teacher_log_probs (feedback-conditioned self-teacher). "
        "Ensure the SDPO teacher pipeline is enabled for this method."
    )
    assert teacher_ent is not None, (
        "SRPO requires teacher_entropy for entropy-aware dynamic weighting."
    )

    beta = float(_sd_cfg(config, "srpo_beta", 1.0))
    alpha = float(_sd_cfg(config, "alpha", 0.5))
    is_clip = float(_sd_cfg(config, "is_clip", 2.0))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    valid_mask = response_mask.to(log_prob.dtype)

    # ---- Sample-level routing: c_i from per-rollout advantage sign --------
    rollout_token_count = valid_mask.sum(dim=1).clamp(min=1.0)
    rollout_adv = (advantages * valid_mask).sum(dim=1) / rollout_token_count
    correct_per_rollout = (rollout_adv > 0).to(log_prob.dtype)         # c_i
    sdpo_per_rollout = 1.0 - correct_per_rollout                        # z^SDPO_i (m_i = 1)
    grpo_per_rollout = 1.0 - sdpo_per_rollout                           # z^GRPO_i

    grpo_mask = grpo_per_rollout.unsqueeze(1) * valid_mask
    sdpo_mask = sdpo_per_rollout.unsqueeze(1) * valid_mask

    # ---- GRPO branch: dual-clip PPO surrogate, per-token (no reduction) ---
    negative_approx_kl = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = negative_approx_kl.exp()
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    grpo_per_token = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        grpo_per_token = grpo_per_token * rollout_is_weights

    # Diagnostics on the GRPO branch only (clipping is meaningless for SDPO tokens).
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    grpo_diag_mask = grpo_mask if grpo_mask.sum() > 0 else response_mask
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), grpo_diag_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        (torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float()),
        grpo_diag_mask)

    # ---- SDPO branch: logit-level distillation, per-token (no reduction) --
    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="student")
        per_token_kl = _sd_generalized_jsd(student_distill, teacher_distill, alpha)
    else:
        if alpha != 1.0:
            raise AssertionError("Non-full-logit SRPO/SDPO only supports alpha=1.0 (k1 reverse-KL estimator).")
        log_ratio = log_prob - teacher_lp.detach()
        per_token_kl = log_ratio.detach() * log_prob

    is_ratio = (log_prob - old_log_prob).detach().clamp(-20, 20).exp().clamp(max=is_clip)
    sdpo_per_token = per_token_kl * is_ratio
    if rollout_is_weights is not None:
        sdpo_per_token = sdpo_per_token * rollout_is_weights

    # ---- Entropy-aware Dynamic Weighting on the SDPO branch ---------------
    # w̃_{i,t} = exp(-β · H_{i,t}); normalize so the mean weight over routed
    # tokens is 1, preserving the overall loss scale (paper §3.2).
    with torch.no_grad():
        teacher_ent_d = teacher_ent.detach().to(log_prob.dtype)
        log_w = -beta * teacher_ent_d
        # Numerically stable exp: subtract the max log-weight over routed tokens.
        # Tokens outside the SDPO route get -1e30 so they don't influence the
        # max; the (log_w - max) difference is then clamped to avoid +inf when
        # no SDPO tokens exist in the batch.
        log_w_for_max = torch.where(
            sdpo_mask > 0, log_w, torch.full_like(log_w, -1.0e30)
        )
        log_w_max = log_w_for_max.amax()
        w_unnorm = ((log_w - log_w_max).clamp(max=50.0)).exp() * sdpo_mask
        n_sdpo_tokens = sdpo_mask.sum().clamp(min=1.0)
        w_mean = w_unnorm.sum() / n_sdpo_tokens
        dw_weights = w_unnorm / w_mean.clamp(min=1e-12)

    sdpo_per_token = sdpo_per_token * dw_weights

    # ---- Final routed loss with unified denominator -----------------------
    grpo_term = (grpo_per_token * grpo_mask).sum()
    sdpo_term = (sdpo_per_token * sdpo_mask).sum()
    denom = (grpo_mask.sum() + sdpo_mask.sum()).clamp(min=1.0)
    loss = (grpo_term + sdpo_term) / denom

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "srpo/correct_rollout_frac": correct_per_rollout.mean().detach().item(),
        "srpo/sdpo_token_frac": (sdpo_mask.sum() / valid_mask.sum().clamp(min=1.0)).detach().item(),
        "srpo/grpo_loss_term": (grpo_term / denom).detach().item(),
        "srpo/sdpo_loss_term": (sdpo_term / denom).detach().item(),
        "srpo/dw_beta": beta,
        "srpo/dw_weight_mean": w_mean.detach().item(),
        "srpo/is_ratio": is_ratio.mean().detach().item(),
        "distill_alpha": alpha,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill)}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, teacher_ent))
    return loss, metrics
