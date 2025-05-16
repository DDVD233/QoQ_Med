# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
implement PPO
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple
from ..utils import torch_functional as VF
import torch.nn.functional as F
import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from sklearn.cluster import KMeans


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None:
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


EPS_DEFAULT: float = 1e-6

# Per‑domain question history ------------------------------------------------ #
#   domain_qstats[dom] = {
#       "vectors": List[np.ndarray]   # shape = (Q, R)
#       "q_ids":   List[int],        # question ids in same order as vectors
#       "count":   int,              # #questions accumulated so far
#   }
# --------------------------------------------------------------------------- #
domain_qstats: Dict[Any, Dict[str, Any]] = defaultdict(lambda: {
    "vectors": [],
    "q_ids":   [],
    "count":   0,
})

global_running_stats: Dict[str, int] = {"q_count": 0}

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def _quantile_safe(x: torch.Tensor, q: float, eps: float) -> torch.Tensor:
    if torch.allclose(x, torch.zeros_like(x)):
        return torch.tensor(eps, dtype=x.dtype, device=x.device)
    return torch.quantile(x, q).detach().clamp_min(eps)


def _select_k_elbow(vals: np.ndarray, k_max: int = 10, tol: float = 0.10) -> int:
    """k‑means elbow pick on multi‑dimensional points."""
    unique_cnt = len(np.unique(vals, axis=0))
    k_cap      = min(k_max, unique_cnt)
    ks         = range(1, k_cap + 1)
    inertias   = [KMeans(n_clusters=k, n_init="auto", random_state=0).fit(vals).inertia_ for k in ks]
    if len(inertias) == 1:
        return 1
    drops = np.diff(inertias) * -1.0
    for i in range(1, len(drops)):
        if drops[i] < tol * drops[i - 1]:
            return i + 1
    return ks[-1]


def _cluster_info_question(vectors: List[np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """K‑means on question‑level vectors.

    Returns
    -------
    mu_d        : float   – inverse‑cluster‑size weighted mean of the centroid means
    assignments : (Q,)    – cluster index for each question vector
    counts      : (k,)    – cluster sizes
    centroids   : (k,R)   – cluster centroid vectors
    """
    if len(vectors) == 0:
        return 0.0, np.empty(0, int), np.empty(0), np.empty((0, 0))

    X = np.stack(vectors, axis=0)            # (Q,R) – R inferred from data
    k_opt = _select_k_elbow(X, k_max=10)
    km    = KMeans(n_clusters=k_opt, n_init="auto", random_state=0).fit(X)

    centroids   = km.cluster_centers_        # (k,R)
    assignments = km.labels_                 # (Q,)
    _, counts   = np.unique(assignments, return_counts=True)
    counts      = counts.astype(float)

    centroid_means = centroids.mean(axis=1)  # (k,)
    weights        = 1.0 / counts
    mu_d           = float((weights * centroid_means).sum() / weights.sum())

    # Debug ------------------------------------------------------------- #
    print(
        f"[KMEANS‑Q] k={k_opt} | centroid_means="
        f"[{', '.join(f'{m:.3f}' for m in centroid_means)}] | counts={counts.tolist()} | μ_d={mu_d:.3f}"
    )

    return mu_d, assignments, counts, centroids

# --------------------------------------------------------------------------- #
#  Main advantage routine                                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()

def compute_drpo_outcome_advantage(
    token_level_rewards: torch.Tensor,      # (B,L)
    response_mask:      torch.Tensor,       # (B,L)
    index:              np.ndarray[str],         # (B,) question ids
    domain_info:        np.ndarray,         # (B,) domain ids
    log_probs:          torch.Tensor,       # (B,L)
    ref_log_probs:      torch.Tensor,       # (B,L)
    eps: float = EPS_DEFAULT,
    kl_q: float = 0.75,
):
    """DRPO with question‑level clustering (no fixed R)."""

    B, L = token_level_rewards.shape

    # 1) raw rollout‑level rewards -------------------------------------- #
    raw_scores = token_level_rewards.sum(dim=-1)                          # (B,)

    # 2) collect rollouts per question for this mini‑batch -------------- #
    q2rollouts: Dict[str, List[float]] = defaultdict(list)
    q2domain:   Dict[str, Any]         = {}
    for i in range(B):
        qid: str = index[i]
        q2rollouts[qid].append(raw_scores[i].item())
        q2domain[qid] = domain_info[i]

    # ensure consistent rollout count ----------------------------------- #
    rollout_lens = {len(v) for v in q2rollouts.values()}
    assert len(rollout_lens) == 1, "Inconsistent rollout counts per question in batch!"

    # build vector per question ----------------------------------------- #
    q_vectors = {qid: np.asarray(v, dtype=np.float32) for qid, v in q2rollouts.items()}

    # 3) update per‑domain question history ----------------------------- #
    for qid, vec in q_vectors.items():
        dom = q2domain[qid]
        dstat = domain_qstats[dom]
        dstat["vectors"].append(vec)
        dstat["q_ids"].append(qid)
        dstat["count"] += 1
        global_running_stats["q_count"] += 1

    # 4) GRPO normalisation (within‑question) --------------------------- #
    scores = raw_scores.clone()
    id2mean = {qid: torch.mean(torch.tensor(v)) for qid, v in q2rollouts.items()}
    id2std  = {qid: torch.std (torch.tensor(v)) for qid, v in q2rollouts.items()}
    for i in range(B):
        qid: str = index[i]
        scores[i] = (scores[i] - id2mean[qid]) / (id2std[qid] + eps)
    before_scale_score = scores.clone()

    # 5) Domain‑wise question clustering -------------------------------- #
    domain_cluster_cache: Dict[Any, Dict[str, Any]] = {}
    for dom, dstat in domain_qstats.items():
        if dstat["count"] == 0:
            continue
        mu_d, assign, counts, centroids = _cluster_info_question(dstat["vectors"])
        domain_cluster_cache[dom] = {
            "mu_d":      mu_d,
            "assign":    assign,
            "counts":    counts,
            "centroids": centroids,
            "q_ids":     dstat["q_ids"],
        }

    # 6) Apply scaling --------------------------------------------------- #
    scaling_factors: List[float] = []
    for i in range(B):
        qid: str  = index[i]
        dom  = q2domain[qid]
        cache = domain_cluster_cache[dom]

        # map qid → cluster idx ---------------------------------------- #
        q_idx       = cache["q_ids"].index(qid)
        cluster_idx = cache["assign"][q_idx]

        N_d  = float(domain_qstats[dom]["count"])
        mu_d = cache["mu_d"]
        T_d  = max(math.sqrt(N_d) * mu_d, eps)

        N_c  = float(cache["counts"][cluster_idx])
        mu_c = float(cache["centroids"][cluster_idx].mean())

        factor = T_d * math.sqrt(N_c) * mu_c
        scaling_factors.append(factor)
        scores[i] = scores[i] / factor

    # divide scores by std of scores
    scores_std = torch.std(scores)
    scores = scores / (scores_std + eps)

    # Debug report -------------------------------------------------------- #
    print("--------------Hierarchical scaling report--------------")
    dom2scale: Dict[Any, List[torch.Tensor]] = defaultdict(list)
    for i in range(B):
        dom2scale[domain_info[i]].append(scores[i] / (before_scale_score[i] + eps))
    for dom, lst in dom2scale.items():
        avg_sf = torch.mean(torch.stack(lst)).item()
        print(f"[HDRPO] domain = {dom:<15} | mean overall scale = {avg_sf:6.3f}")

    # Print global reward mean
    print(f"[HDRPO] global reward mean = {torch.mean(scores):.3f}")

    # 7) KL‑aware damping ---------------------------------------------- #
    # kl_tok     = compute_kl(log_probs, ref_log_probs, "low_var_kl")
    # kl_tok     *= response_mask
    # kl_rollout = kl_tok.sum(dim=-1)
    #
    # z_abs = scores.abs() * kl_rollout
    # t     = _quantile_safe(z_abs, kl_q, eps)
    # m     = t / (z_abs + t)
    # scores = m * scores

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the policy loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy

    """
    negative_approx_kl = log_probs - old_log_probs
    # clamp the ratio before exp to avoid nan
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(negative_approx_kl)
    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * clip_ratio_dual

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    pg_clipfrac_higher = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    pg_clipfrac_lower = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = VF.masked_mean(final_pg_loss, response_mask)
    pg_clipfrac_higher = VF.masked_mean(pg_clipfrac_higher, response_mask)
    pg_clipfrac_lower = VF.masked_mean(pg_clipfrac_lower, response_mask)
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)
    return final_pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    cliprange_value: float,
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_loss1, vf_loss2), action_mask)  # clip if vf_loss1 < vf_loss2
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), action_mask)
    return vf_loss, vf_clipfrac


def compute_kl(log_probs: torch.FloatTensor, ref_log_probs: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        kl = ref_log_probs - log_probs
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
