"""Verify selector gradients, the distillation denominator and the packed Qwen forward."""
from types import SimpleNamespace
import torch
from verl.trainer.ppo.core_algos import _opsd_distill_term, _random_gate_matching_counts

torch.manual_seed(42)
eligible = torch.tensor([[1, 1, 0, 1, 1, 1], [0]*6, [1]*6, [1, 1, 0, 0, 0, 0]]).bool()
gate = torch.tensor([[1, 0, 1, 0, 0, 1], [1]*6, [0]*6, [1]*6]).bool()
draws = torch.stack([_random_gate_matching_counts(gate, eligible) for _ in range(500)])
assert not (draws & ~eligible).any()
assert (draws.sum(-1) == (eligible & gate).sum(-1)).all()
assert (draws[:, 0].any(0) == eligible[0]).all()
assert draws[:, 0].unique(dim=0).size(0) > 1

mask = torch.ones(2, 12, dtype=torch.long)
mask[0, -2:] = 0
gold = torch.tensor([.85, .10]).log().expand(2, 12, 2).clone()
cf = gold.clone()
cf[:, ::2] = torch.tensor([.10, .85]).log()
for mode in ["goldside_js_hard", "goldside_js_random_matched"]:
    raw = torch.tensor([.45, .40]).log().expand_as(gold).clone().requires_grad_()
    config = SimpleNamespace(policy_loss=dict(
        distill_key_ablation_gate=True, distill_key_ablation_gate_mode=mode,
        distill_key_ablation_tau=.1, beta=1.0, jsd_token_clip=0,
        top_k_distill=2, full_logit_distill=True))
    loss, metrics = _opsd_distill_term(raw[..., 0], mask, config, None,
        teacher_log_probs=gold[..., 0], teacher_keyabl_log_probs=cf[..., 0],
        teacher_topk_log_probs=gold, teacher_keyabl_topk_log_probs=cf,
        student_topk_log_probs=raw, distill_key_ablation_sample_mask=torch.tensor([1, 0]))
    loss.backward()
    active = raw.grad.abs().sum(-1) > 0
    assert active.sum(1).tolist() == [5, 0] and not (active & ~mask.bool()).any()
    assert abs(metrics["distill_keyabl_gate_active_ratio"] - .5) < 1e-6
    assert torch.isfinite(loss) and loss.item() > 0
    print(mode, "loss", loss.item(), "gradient positions per row", active.sum(1).tolist())

# Use variable KL values and CF budgets of 2, 0 and 4 on certified rows.
# Uncertified rows and padding have large scores to expose selection leaks.
valid = torch.tensor([[1, 1, 1, 1, 0]] * 4, dtype=torch.bool)
certified = torch.tensor([True, False, True, True])
eligible = valid & certified[:, None]
gold_probs = torch.tensor([.6, .3, .1], dtype=torch.float64).expand(4, 5, 3)
cf_top = gold_probs[..., :2].clone()
flip = gold_probs.new_tensor([.05, .90])
cf_top[0, [0, 1, 4]] = flip
cf_top[1] = flip
cf_top[3] = flip
probs = gold_probs.new_tensor([[.59, .30, .11], [.74, .2599, .0001], [.43, .23, .34],
                      [.25, .15, .60], [.01, .01, .98]]).expand(4, 5, 3)
kl = (probs * (probs.log() - gold_probs.log())).sum(-1)
# Large importance weights on other positions must not change the ranking.
weights = torch.tensor([10000., 100., 1., 1., 1.]).expand_as(valid)
for mode in ("goldside_js_random_matched", "goldside_js_kl_matched", "certified_dense"):
    dense = mode == "certified_dense"
    config = SimpleNamespace(policy_loss=dict(
        distill_key_ablation_gate=not dense, distill_key_ablation_gate_mode=mode,
        distill_certified_only=dense, distill_key_ablation_tau=.1, beta=1., jsd_token_clip=0.,
        full_logit_distill=True, top_k_distill=2))
    student = probs[..., :2].log().clone().requires_grad_()
    inputs = dict(teacher_log_probs=gold_probs[..., 0].log(), teacher_topk_log_probs=gold_probs[..., :2].log(),
                  student_topk_log_probs=student, distill_key_ablation_sample_mask=certified)
    if not dense:
        inputs.update(teacher_keyabl_topk_log_probs=cf_top.log(), teacher_keyabl_log_probs=cf_top[..., 0].log())
    loss, _ = _opsd_distill_term(student[..., 0], valid, config, weights, **inputs)
    loss.backward()
    active = student.grad.abs().sum(-1) > 0
    assert not (active & ~eligible).any()
    assert active.sum(1).tolist() == ([4, 0, 4, 4] if dense else [2, 0, 0, 4])
    if mode == "goldside_js_kl_matched":
        assert active[0].tolist() == [False, False, True, True, False]
    torch.testing.assert_close(loss.detach(), (weights * kl * active).sum() / eligible.sum(),
                               atol=1e-10, rtol=1e-10)
    print(mode, "verified gradients and eligible-token denominator")

# The Qwen3 memory-saving path must preserve both token probabilities and gradients.
from omegaconf import OmegaConf
import sys
if '--qwen2' in sys.argv:
    from transformers import Qwen2Config as Config, Qwen2ForCausalLM as Model
else:
    from transformers import Qwen3Config as Config, Qwen3ForCausalLM as Model
from verl.workers.actor.dp_actor import DataParallelPPOActor

actor = object.__new__(DataParallelPPOActor)
actor.device_name = "cuda" if torch.cuda.is_available() else "cpu"
actor.param_dtype = torch.bfloat16
actor.use_prefix_grouper = actor.use_ulysses_sp = actor.use_fused_kernels = False
actor.use_remove_padding = True
actor.config = OmegaConf.create({"response_only_lm_head": False})
actor.actor_module = Model(Config(vocab_size=128, hidden_size=32, head_dim=16,
    intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).to(actor.device_name)
ids = torch.randint(1, 128, (2, 12), device=actor.device_name)
attention = torch.ones_like(ids)
attention[0, :2] = attention[0, -1] = 0
batch = dict(input_ids=ids, responses=ids[:, -4:], attention_mask=attention,
             position_ids=(attention.cumsum(-1)-1).clamp_min(0), response_mask=attention[:, -4:])
support = torch.randint(0, 128, (2, 4, 5), device=actor.device_name)
results = []
for enabled in (False, True):
    actor.config.response_only_lm_head = enabled
    actor.actor_module.zero_grad(set_to_none=True)
    out = actor._forward_micro_batch(batch, temperature=1., topk_indices=support)
    ((out["log_probs"] + out["topk_log_probs"].mean(-1)) * batch["response_mask"]).sum().backward()
    results.append(([out[k].detach().float() for k in ("log_probs", "topk_log_probs")],
                    [p.grad.detach().float().clone() for p in actor.actor_module.parameters()]))
for ref, actual in zip(results[0][0], results[1][0]):
    torch.testing.assert_close(actual, ref, atol=.04, rtol=.005)
for ref, actual in zip(results[0][1], results[1][1]):
    torch.testing.assert_close(actual, ref, atol=.035, rtol=.02)
print(Model.__name__, "packed response logits and gradients match full projection")
