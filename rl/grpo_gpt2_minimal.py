# pip install torch==2.* transformers==4.* accelerate==1.* numpy
import os, math, copy, random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 0) Config
# -----------------------------
MODEL_NAME = "gpt2"            # small, CPU-friendly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4                  # prompts per "batch"
GEN_MAX_NEW_TOKENS = 64
TEMPERATURE = 0.9
TOP_P = 0.95
LR = 1e-5
CLIP_EPS = 0.2
EPOCHS_PER_BATCH = 2            # PPO epochs per collected batch
OUTER_STEPS = 5                 # how many collect+update cycles
KL_BETA = 0.02                  # small KL to reference model (set 0.0 to disable)
LOGPROB_DIFF_CLAMP = 20.0       # stabilize exp(lp_new - lp_old)
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# -----------------------------
# 1) Example prompts
# -----------------------------
PROMPTS = [
    "Write a one-sentence explanation of overfitting.",
    "List three prime numbers under 20 in a single short sentence.",
    "Explain what reinforcement learning from human feedback (RLHF) is in one sentence.",
    "Give a single-sentence haiku-like line about the moon (keep it short).",
    "In one sentence, say why unit tests are useful.",
    "Answer yes/no: Is 37 a prime number? Provide only yes or no.",
    "End your reply with a period and keep it under 15 words: What is entropy?",
    "Name exactly three fruits, comma-separated, then stop.",
]

# -----------------------------
# 2) Tiny heuristic reward
#    (prompt-aware, but simple)
# -----------------------------
def reward_fn(prompt: str, text: str) -> float:
    """
    Small, fast heuristics to make the signal non-trivial but stable.
    You can customize per-prompt rules below.
    Returns a scalar reward (float).
    """
    t = text.strip()

    # generic bonuses
    r = 0.0
    if len(t) > 0: r += 0.2
    if t.endswith("."): r += 0.2
    if len(t.split()) <= 20: r += 0.2  # brevity bonus
    if "\n" not in t: r += 0.1         # single-line bonus

    # prompt-specific nudges
    if "prime numbers" in prompt.lower():
        # simple check: contains three primes under 20 (from set)
        primes = {"2","3","5","7","11","13","17","19"}
        words = {w.strip(",. ").lower() for w in t.split()}
        count = sum(1 for w in words if w in primes)
        # reward grows with count up to 3
        r += min(count, 3) * 0.2

    if "Answer yes/no" in prompt:
        low = t.lower()
        if low in {"yes", "no", "yes.", "no."}:
            r += 0.6
        else:
            r -= 0.2

    if "exactly three fruits" in prompt.lower():
        # crude comma-separated count
        parts = [p.strip() for p in t.split(",")]
        if len(parts) == 3:
            r += 0.6
        elif len(parts) > 0:
            r += 0.2

    if "haiku" in prompt.lower():
        # prefer ~5-10 words
        n = len(t.split())
        r += 0.4 if 5 <= n <= 10 else 0.0

    if "End your reply with a period" in prompt:
        if t.endswith("."):
            r += 0.4
        else:
            r -= 0.2

    # small penalty for being too long
    if len(t.split()) > 30:
        r -= 0.3

    return float(r)

# -----------------------------
# 3) Utilities
# -----------------------------
def prepare_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 has no pad token; use EOS for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only models expect left padding for sampling

    policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    policy.to(DEVICE)
    policy.train()

    # Reference (for optional KL) is a frozen copy of the *initial* policy
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad = False

    return tokenizer, policy, ref_model

@torch.no_grad()
def generate(policy, tokenizer, prompts):
    """
    Generate continuations and return:
      - responses (strings)
      - ids_all (concatenated prompt+response ids)
      - attn_all
      - prompt_lens (for slicing response part)
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    prompt_input_ids = enc.input_ids
    prompt_attn = enc.attention_mask
    context_len = prompt_input_ids.size(1)
    prompt_lens = [context_len] * prompt_input_ids.size(0)

    gen_out = policy.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attn,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Decode just the new tokens for display, but also return full
    responses = []
    for i in range(len(prompts)):
        full = gen_out[i]
        resp_ids = full[context_len:]
        responses.append(tokenizer.decode(resp_ids, skip_special_tokens=True))

    attn_all = (gen_out != tokenizer.pad_token_id).long()
    return responses, gen_out, attn_all, prompt_lens

def seq_logprob_for_response(model, input_ids, attention_mask, prompt_len):
    """
    Compute log p(response | prompt) by summing token logprobs on response positions only.
    For a single sample (no batch). Returns scalar tensor.
    """
    # Shift for next-token prediction:
    # logits for position t predict token at t+1
    outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    logits = outputs.logits.squeeze(0)  # [T, V]

    log_probs = torch.log_softmax(logits, dim=-1)  # [T, V]
    # Response tokens are those after the prompt_len
    # For token at position t-1, model predicts token at t
    start = prompt_len  # first response token index in input_ids
    # We sum logprob of tokens at positions start .. T-1, predicted by logits at positions start-1 .. T-2
    # So slice logits[ start-1 : T-1 ] vs tokens[ start : T ]
    if start <= 0:  # safety
        start = 1
    target_tokens = input_ids[start:]                    # [R]
    pred_log_probs = log_probs[start-1: -1]              # [R, V]
    tok_lp = pred_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)  # [R]
    return tok_lp.sum()

def batch_logprobs(model, input_ids, attention_mask, prompt_lens):
    """
    Vectorized helper to collect log p(response|prompt) for each example.
    Falls back to per-sample loop for clarity.
    """
    out = []
    for i in range(input_ids.size(0)):
        out.append(seq_logprob_for_response(model, input_ids[i], attention_mask[i], prompt_lens[i]))
    return torch.stack(out)  # [B]

@torch.no_grad()
def batch_ref_kl(model_ref, input_ids, attention_mask, prompt_lens, model_cand):
    """
    Compute KL( cand || ref ) over response tokens for each sample.
    """
    kl_vals = []
    for i in range(input_ids.size(0)):
        inp = input_ids[i].unsqueeze(0)
        attn = attention_mask[i].unsqueeze(0)
        T = inp.size(1)
        start = max(1, prompt_lens[i])  # avoid 0

        # candidate
        cand_logits = model_cand(input_ids=inp, attention_mask=attn).logits  # [1,T,V]
        cand_logp = torch.log_softmax(cand_logits, dim=-1)[:, start-1:-1, :]  # [1,R,V]

        # reference
        ref_logits = model_ref(input_ids=inp, attention_mask=attn).logits
        ref_logp = torch.log_softmax(ref_logits, dim=-1)[:, start-1:-1, :]    # [1,R,V]

        # use teacher-forced targets to define the response token support
        targets = inp[:, start:]  # [1,R]
        # KL per position using the distribution rows (no masking here, already aligned)
        # KL(p||q) = sum p * (log p - log q) over vocab. We approximate by
        # evaluating only at the realized token (token-level cross-entropy diff),
        # which is a light proxy (not full distribution KL) to keep it cheap.
        # If you want exact KL, compute (p* (logp - logq)).sum(-1).
        cand_lp_tok = cand_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1,R]
        ref_lp_tok  = ref_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [1,R]
        kl_tokenwise = (cand_lp_tok - ref_lp_tok).neg()  # -log q + log p ~ KL proxy
        kl_vals.append(kl_tokenwise.mean().detach().cpu())
    return torch.stack(kl_vals).to(DEVICE)  # [B]

def z_score(x: torch.Tensor, eps=1e-8):
    return (x - x.mean()) / (x.std(unbiased=False) + eps)

# -----------------------------
# 4) Main GRPO loop
# -----------------------------
def main():
    tokenizer, policy, ref_model = prepare_models()
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    print(f"Device: {DEVICE}, model: {MODEL_NAME}")
    print("Starting GRPO…\n")

    for outer in range(OUTER_STEPS):
        # === 4.1 Collect a batch ===
        prompts = random.sample(PROMPTS, BATCH_SIZE)
        # Freeze old policy snapshot for the ratio
        old_policy = copy.deepcopy(policy).eval()
        for p in old_policy.parameters(): p.requires_grad = False

        # Generate with *current* policy
        policy.eval()
        responses, ids_all, attn_all, prompt_lens = generate(policy, tokenizer, prompts)
        policy.train()

        # Show samples
        print(f"\n=== Collect step {outer+1}/{OUTER_STEPS} ===")
        for i, (pr, rsp) in enumerate(zip(prompts, responses)):
            print(f"[{i}] Prompt: {pr}")
            print(f"    Response: {rsp}")

        # Compute rewards
        rewards = []
        for pr, rsp in zip(prompts, responses):
            rewards.append(reward_fn(pr, rsp))
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

        # Optional: small KL penalty (candidate vs ref) over response tokens
        if KL_BETA > 0.0:
            kl_vals = batch_ref_kl(ref_model, ids_all, attn_all, prompt_lens, policy)
            rewards = rewards - KL_BETA * kl_vals

        # Normalize rewards => pseudo-advantages (GRPO)
        adv = z_score(rewards)

        # Compute log p_old(y|x) under frozen old policy
        with torch.no_grad():
            lp_old = batch_logprobs(old_policy, ids_all, attn_all, prompt_lens)  # [B]

        # === 4.2 Optimize (PPO-style) ===
        for e in range(EPOCHS_PER_BATCH):
            # Recompute log p_new(y|x) under current policy
            lp_new = batch_logprobs(policy, ids_all, attn_all, prompt_lens)  # [B]
            log_ratio = torch.clamp(lp_new - lp_old, -LOGPROB_DIFF_CLAMP, LOGPROB_DIFF_CLAMP)
            ratio = torch.exp(log_ratio)  # [B]

            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
            loss = -torch.mean(torch.minimum(unclipped, clipped))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (lp_old - lp_new).mean().abs().item()

            print(f"  Epoch {e+1}/{EPOCHS_PER_BATCH} | Loss {loss.item():.4f} | "
                  f"Reward μ {rewards.mean().item():+.3f} | Adv std {adv.std().item():.3f} | "
                  f"KL~ {approx_kl:.4f}")

    print("\nDone. Try generating again to see shifted behavior!\n")
    # quick demo
    demo = ["List three prime numbers under 20 in a single short sentence."]
    policy.eval()
    rsp, *_ = generate(policy, tokenizer, demo)
    print("Demo:", rsp[0])

if __name__ == "__main__":
    main()
