# dpo_gpt2_minimal.py
# Minimal Direct Preference Optimization (DPO) training with GPT-2.
# pip install torch transformers

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-5
BETA = 0.1   # DPO beta (scales the preference strength)
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------------------------------
# 1) Tiny toy preference dataset
#    Each example: (prompt, chosen, rejected)
# -------------------------------------------------
DATA = [
    (
        "Question: Is 37 a prime number?\nAnswer:",
        " Yes.",           # chosen
        " No.",            # rejected
    ),
    (
        "List three prime numbers under 20.\nAnswer:",
        " 2, 3, 5.",
        " 4, 6, 8.",
    ),
    (
        "Name exactly three fruits.\nAnswer:",
        " apple, banana, orange.",
        " apple, banana, orange, mango.",
    ),
    (
        "Explain what overfitting is in one short sentence.\nAnswer:",
        " Overfitting is when a model learns noise in the training data.",
        " Overfitting is when a model perfectly fits both data and noise forever.",
    ),
]

# -------------------------------------------------
# 2) Helpers: log p(completion | prompt) under GPT-2
# -------------------------------------------------
def seq_logprob_conditional(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    """
    Compute log p(completion | prompt) as sum of token logprobs for completion tokens,
    conditioned on the full prompt.

    Returns a scalar tensor (on DEVICE).
    """
    # Encode prompt and full sequence (prompt + completion)
    enc_prompt = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    enc_full = tokenizer(prompt + completion, add_special_tokens=False, return_tensors="pt")

    input_ids = enc_full["input_ids"].to(DEVICE)    # [1, T]
    attention_mask = torch.ones_like(input_ids).to(DEVICE)
    prompt_len = enc_prompt["input_ids"].shape[1]   # P

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [1, T, V]

    logits = logits[0]          # [T, V]
    input_ids = input_ids[0]    # [T]

    # Completion tokens indices: P .. T-1 (inclusive)
    # For next-token prediction:
    #  - logits at position t predict token at t+1
    #  - so to predict token at index i, we use logits at index i-1
    start = prompt_len
    if start <= 0:
        start = 1
    target_tokens = input_ids[start:]          # [R]
    logits_slice = logits[start-1:-1, :]       # [R, V]

    log_probs = torch.log_softmax(logits_slice, dim=-1)  # [R, V]
    tok_logp = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)  # [R]
    return tok_logp.sum()  # scalar

# -------------------------------------------------
# 3) Load tokenizer, policy model, and reference model
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # causal LM-friendly

policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
policy.train()

# Reference is frozen copy (often the SFT model)
reference = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
reference.eval()
for p in reference.parameters():
    p.requires_grad = False

optimizer = optim.Adam(policy.parameters(), lr=LR)

# -------------------------------------------------
# 4) DPO training loop
# -------------------------------------------------
def dpo_batch_loss(batch):
    """
    batch: list of (prompt, chosen, rejected)
    returns scalar loss tensor
    """
    # We'll accumulate for the batch
    g_list = []

    for (prompt, chosen, rejected) in batch:
        # log p under current policy
        logp_pi_chosen = seq_logprob_conditional(policy, tokenizer, prompt, chosen)
        logp_pi_rejected = seq_logprob_conditional(policy, tokenizer, prompt, rejected)

        # log p under reference (frozen)
        with torch.no_grad():
            logp_ref_chosen = seq_logprob_conditional(reference, tokenizer, prompt, chosen)
            logp_ref_rejected = seq_logprob_conditional(reference, tokenizer, prompt, rejected)

        # DPO logit for this preference pair
        # g = β[(logπθ(y+) - logπref(y+)) - (logπθ(y−) - logπref(y−))]
        g = BETA * (
            (logp_pi_chosen - logp_ref_chosen) -
            (logp_pi_rejected - logp_ref_rejected)
        )
        g_list.append(g)

    g_tensor = torch.stack(g_list)  # [B]
    # DPO loss = - E[ log σ(g) ]
    loss = -torch.mean(torch.log(torch.sigmoid(g_tensor)))
    return loss

print(f"Device: {DEVICE}")
print("Starting DPO training...\n")

for epoch in range(EPOCHS):
    # simple full-batch or small-batch training over DATA
    total_loss = 0.0
    n_steps = 0

    # mini-batching over DATA
    for i in range(0, len(DATA), BATCH_SIZE):
        batch = DATA[i:i+BATCH_SIZE]

        optimizer.zero_grad()
        loss = dpo_batch_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / max(n_steps,1):.4f}")

# -------------------------------------------------
# 5) Quick qualitative check
# -------------------------------------------------
policy.eval()
demo_prompt = "Question: Is 37 a prime number?\nAnswer:"
inputs = tokenizer(demo_prompt, return_tensors="pt").to(DEVICE)
out = policy.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print("\nDemo completion:")
print(tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
