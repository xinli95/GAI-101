# ppo_gpt2_minimal.py
# Minimal PPO fine-tuning for GPT-2 with a tiny heuristic reward.
# pip install torch transformers numpy

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------
# Config
# --------------------
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 4                # prompts per rollout
MAX_NEW_TOKENS = 48
TEMPERATURE = 0.9
TOP_P = 0.95

OUTER_STEPS = 6               # rollout+opt cycles
PPO_EPOCHS = 2                # epochs over collected batch
LR = 1e-5
CLIP_EPS = 0.2
GAMMA = 0.99
LAMBDA = 0.95
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 1.0

# --------------------
# Example prompts
# --------------------
PROMPTS = [
    "Write a one-sentence explanation of overfitting.",
    "List three prime numbers under 20 in a single short sentence.",
    "Explain what reinforcement learning from human feedback (RLHF) is in one sentence.",
    "Answer yes/no only: Is 37 a prime number?",
    "End your reply with a period and keep it under 15 words: What is entropy?",
    "Name exactly three fruits, comma-separated, then stop.",
]

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# --------------------
# Heuristic reward
# --------------------
def reward_fn(prompt: str, text: str) -> float:
    t = text.strip()
    r = 0.0
    if len(t) > 0: r += 0.2
    if t.endswith("."): r += 0.2
    if len(t.split()) <= 20: r += 0.2
    if "\n" not in t: r += 0.1

    if "prime numbers" in prompt.lower():
        primes = {"2","3","5","7","11","13","17","19"}
        words = {w.strip(",. ").lower() for w in t.split()}
        r += min(sum(w in primes for w in words), 3) * 0.2

    if "Answer yes/no only" in prompt:
        if t.lower() in {"yes", "no", "yes.", "no."}: r += 0.6
        else: r -= 0.2

    if "exactly three fruits" in prompt.lower():
        parts = [p.strip() for p in t.split(",")]
        if len(parts) == 3: r += 0.6
        elif len(parts) > 0: r += 0.2

    if "End your reply with a period" in prompt:
        r += 0.4 if t.endswith(".") else -0.2

    if len(t.split()) > 30: r -= 0.3
    return float(r)

# --------------------
# Value head (shared backbone)
# --------------------
class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):  # [..., H]
        return self.value(hidden_states).squeeze(-1)  # [...]

# --------------------
# Helpers
# --------------------
@torch.no_grad()
def generate(policy, tokenizer, prompts):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    prompt_input_ids = enc.input_ids
    prompt_attn = enc.attention_mask
    context_len = prompt_input_ids.size(1)
    prompt_lens = [context_len] * prompt_input_ids.size(0)

    gen_out = policy.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attn,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    attn_all = (gen_out != tokenizer.pad_token_id).long()

    responses = []
    for i in range(len(prompts)):
        resp_ids = gen_out[i, context_len:]
        responses.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
    return responses, gen_out, attn_all, prompt_lens

def policy_outputs(model, value_head, input_ids, attention_mask, prompt_lens):
    """
    Compute:
      - per-token logprobs of realized response tokens
      - per-token values
      - per-token entropy
      - mask over response tokens
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    logits = outputs.logits          # [B, T, V]
    hidden = outputs.hidden_states[-1]  # [B, T, H]
    B, T, V = logits.shape

    logprobs_list, values_list, entropy_list, masks_list = [], [], [], []

    for i in range(B):
        start = max(1, prompt_lens[i])
        if start >= T:
            start = T - 1

        logits_r = logits[i, start-1:T-1, :]     # [R, V]
        hidden_r = hidden[i, start-1:T-1, :]     # [R, H]
        targets  = input_ids[i, start:T]         # [R]
        mask_r   = attention_mask[i, start:T]    # [R]

        logp = torch.log_softmax(logits_r, dim=-1)       # [R, V]
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [R]
        vals = value_head(hidden_r)                      # [R]

        probs = torch.softmax(logits_r, dim=-1)
        ent = -(probs * logp).sum(-1)                    # [R]

        logprobs_list.append(tok_logp)
        values_list.append(vals)
        entropy_list.append(ent)
        masks_list.append(mask_r.float())

    logprob_tokens = pad_sequence(logprobs_list, batch_first=True, padding_value=0.0)
    values_r = pad_sequence(values_list, batch_first=True, padding_value=0.0)
    entropy_r = pad_sequence(entropy_list, batch_first=True, padding_value=0.0)
    mask_r = pad_sequence(masks_list, batch_first=True, padding_value=0.0)
    return logprob_tokens, values_r, entropy_r, mask_r

def compute_gae(reward_tokens, values, mask, gamma=0.99, lam=0.95):
    B, R = reward_tokens.shape
    advantages = torch.zeros_like(reward_tokens)
    lastgaelam = torch.zeros(B, device=reward_tokens.device)

    for t in reversed(range(R)):
        next_nonterminal = mask[:, t]
        next_value = torch.zeros(B, device=reward_tokens.device) if t == R-1 else values[:, t+1]
        delta = reward_tokens[:, t] + gamma * next_value * next_nonterminal - values[:, t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        advantages[:, t] = lastgaelam
    returns = advantages + values
    return advantages, returns

def z_score(x, eps=1e-8):
    return (x - x.mean()) / (x.std(unbiased=False) + eps)

# --------------------
# Main
# --------------------
def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # important for causal LM batching

    policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    policy.train()
    hidden_size = policy.config.n_embd
    value_head = ValueHead(hidden_size).to(DEVICE)
    value_head.train()

    params = list(policy.parameters()) + list(value_head.parameters())
    optimizer = optim.Adam(params, lr=LR)

    print(f"Device: {DEVICE} | Model: {MODEL_NAME}")
    print("Starting PPO...\n")

    for outer in range(OUTER_STEPS):
        # ===== Collect rollouts =====
        prompts = random.sample(PROMPTS, BATCH_SIZE)
        policy.eval()
        responses, ids_all, attn_all, prompt_lens = generate(policy, tokenizer, prompts)
        policy.train()

        print(f"\n=== Rollout {outer+1}/{OUTER_STEPS} ===")
        for i, (p, r) in enumerate(zip(prompts, responses)):
            print(f"[{i}] Prompt: {p}")
            print(f"    Resp : {r}")

        # Freeze old policy for ratio
        old_policy = copy.deepcopy(policy).eval()
        for p in old_policy.parameters():
            p.requires_grad = False

        # ---- Get old logprobs and values snapshot ----
        ids_all = ids_all.to(DEVICE)
        attn_all = attn_all.to(DEVICE)
        with torch.no_grad():
            lp_old, v_old, _, mask_r = policy_outputs(
                old_policy, value_head, ids_all, attn_all, prompt_lens
            )

        # ----- Rewards: only final token gets terminal reward -----
        rewards_scalar = torch.tensor(
            [reward_fn(p, r) for p, r in zip(prompts, responses)],
            dtype=torch.float32, device=DEVICE
        )
        B, R = lp_old.shape
        reward_tokens = torch.zeros(B, R, device=DEVICE)
        valid_lengths = mask_r.sum(dim=1).clamp(min=1)
        last_indices = (valid_lengths - 1).long()
        batch_idx = torch.arange(B, device=DEVICE)
        reward_tokens[batch_idx, last_indices] = rewards_scalar

        # ----- Values & advantages -----
        with torch.no_grad():
            adv, ret = compute_gae(reward_tokens, v_old, mask_r, GAMMA, LAMBDA)
            adv = z_score(adv)

        # Flatten per-token tensors
        lp_old_f = lp_old.detach().reshape(-1)
        adv_f = adv.detach().reshape(-1)
        ret_f = ret.detach().reshape(-1)
        mask_f = mask_r.reshape(-1).bool()

        lp_old_f = lp_old_f[mask_f]
        adv_f = adv_f[mask_f]
        ret_f = ret_f[mask_f]
        v_old_f = v_old.reshape(-1)[mask_f]

        # ===== PPO optimize =====
        for epoch in range(PPO_EPOCHS):
            lp_new, v_new, entropy_r, mask_r2 = policy_outputs(
                policy, value_head, ids_all, attn_all, prompt_lens
            )
            lp_new_f = lp_new.reshape(-1)[mask_f]
            v_new_f = v_new.reshape(-1)[mask_f]
            entropy_f = entropy_r.reshape(-1)[mask_f]

            ratio = torch.exp(lp_new_f - lp_old_f)
            pg1 = ratio * adv_f
            pg2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_f
            policy_loss = -torch.mean(torch.min(pg1, pg2))

            v_clipped = v_old_f + torch.clamp(
                v_new_f - v_old_f, -CLIP_EPS, CLIP_EPS
            )
            vf_losses1 = (v_new_f - ret_f) ** 2
            vf_losses2 = (v_clipped - ret_f) ** 2
            value_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

            entropy_bonus = torch.mean(entropy_f)

            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (lp_old_f - lp_new_f).mean().abs().item()
                clipfrac = torch.mean(
                    (torch.abs(ratio - 1.0) > CLIP_EPS).float()
                ).item()

            print(
                f"  Epoch {epoch+1}/{PPO_EPOCHS} | "
                f"Loss {loss.item():.4f} | PG {policy_loss.item():.4f} | "
                f"V {value_loss.item():.4f} | Ent {entropy_bonus.item():.4f} | "
                f"KL~ {approx_kl:.4f} | clipfrac {clipfrac:.3f} | "
                f"R̄ {rewards_scalar.mean().item():+.3f}"
            )

    # quick demo
    policy.eval()
    demo = ["List three prime numbers under 20 in a single short sentence."]
    enc = tokenizer(demo, return_tensors="pt").to(DEVICE)
    out = policy.generate(
        **enc,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("\nDemo:", tokenizer.decode(out[0, enc.input_ids.size(1):], skip_special_tokens=True))


if __name__ == "__main__":
    main()
