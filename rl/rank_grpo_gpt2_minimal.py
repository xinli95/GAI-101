"""
Minimal Rank-GRPO demo using GPT-2 as the language model.

What this script does:
  - Loads GPT-2 and its tokenizer from Hugging Face
  - Builds a small, semi-realistic conversational recommendation dataset:
      * Prompts about movie preferences
      * Two candidate recommendation lists per prompt:
          - One "good" (highly relevant items)
          - One "bad" (irrelevant items)
  - Encodes these into token sequences with explicit rank boundaries
  - Computes per-rank rewards based on relevance (rel_k)
  - Implements a minimal Rank-GRPO loss (rank-level importance ratio + PPO clipping)
  - Runs a few optimization steps and prints the loss

How to run:
  pip install torch transformers
  python rank_grpo_gpt2_demo.py
"""

import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
#  Helper: token logprobs
# =========================

def compute_token_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-wise log p(y_t | x, y_<t) for a causal LM.

    For simplicity, we:
      - run the model once
      - apply log_softmax over vocab
      - gather logprobs corresponding to labels elsewhere

    input_ids: (B, T)
    attention_mask: (B, T)
    Returns: log_probs: (B, T, V)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs


# =========================
#  Rank-GRPO loss
# =========================

def rank_grpo_loss(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,       # (B, G, T)
    attention_mask: torch.Tensor,  # (B, G, T)
    labels: torch.Tensor,          # (B, G, T), -100 for ignored tokens
    logprobs_old: torch.Tensor,    # (B, G, T), token-wise logp under old policy
    rank_ids: torch.Tensor,        # (B, G, T), 0 = non-rank, 1..N for each rank position
    rewards: torch.Tensor,         # (B, G, N), per-rank returns r(x, y^{(k)})
    clip_eps: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Minimal Rank-GRPO loss implementation.

    Key points:
      - We treat each rank k as an "action":
          * rank-level reward r_k
          * rank-level advantage A_k
          * rank-level importance ratio w_k
      - For each rank, we compute a geometric-mean logprob across all tokens
        belonging to that rank under new and old policies.
      - We use a PPO-style clipped objective at rank level.
    """
    device = input_ids.device
    B, G, T = input_ids.shape
    N = rewards.shape[-1]  # number of ranks

    # Flatten batch and group: (B*G, T)
    BG = B * G
    input_ids_flat = input_ids.view(BG, T)
    attention_mask_flat = attention_mask.view(BG, T)
    labels_flat = labels.view(BG, T)
    logprobs_old_flat = logprobs_old.view(BG, T)
    rank_ids_flat = rank_ids.view(BG, T)  # 0..N

    # Compute current token logprobs
    log_probs_flat = compute_token_logprobs(model, input_ids_flat, attention_mask_flat)
    vocab_dim = log_probs_flat.size(-1)

    # For positions where label == -100, we don't care about logprob; set dummy index
    label_mask = labels_flat.ne(-100)  # (BG, T)
    safe_labels = labels_flat.clone()
    safe_labels[~label_mask] = 0  # avoid out-of-range index
    token_logp_new = log_probs_flat.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp_new = token_logp_new * label_mask  # zero out ignored

    # Old token logprobs already provided
    token_logp_old = logprobs_old_flat * label_mask  # (BG, T)

    # Reshape rewards to (BG, N)
    rewards_flat = rewards.view(BG, N)  # (BG, N)

    # --- Step 1: Compute rank-level advantages per sample (per BG) ---

    mean_r = rewards_flat.mean(dim=-1, keepdim=True)     # (BG, 1)
    std_r = rewards_flat.std(dim=-1, keepdim=True) + eps # (BG, 1)
    advantages = (rewards_flat - mean_r) / std_r         # (BG, N)

    # --- Step 2: Loop over ranks and build Rank-GRPO objective ---

    total_loss_terms = []

    for k in range(1, N + 1):
        # Mask for tokens belonging to rank k
        rank_mask = (rank_ids_flat == k) & label_mask  # (BG, T)

        if not rank_mask.any():
            continue

        # Count tokens per (BG) for rank k
        token_counts = rank_mask.sum(dim=-1)  # (BG,)
        valid_bg_mask = token_counts > 0      # (BG,)
        if not valid_bg_mask.any():
            continue

        # Sum logprobs across tokens of this rank
        sum_logp_new = (token_logp_new * rank_mask).sum(dim=-1)  # (BG,)
        sum_logp_old = (token_logp_old * rank_mask).sum(dim=-1)  # (BG,)

        mean_logp_new = torch.zeros_like(sum_logp_new)
        mean_logp_old = torch.zeros_like(sum_logp_old)

        mean_logp_new[valid_bg_mask] = (
            sum_logp_new[valid_bg_mask] / token_counts[valid_bg_mask]
        )
        mean_logp_old[valid_bg_mask] = (
            sum_logp_old[valid_bg_mask] / token_counts[valid_bg_mask]
        )

        # Rank-level importance ratio w_k = exp(mean_logp_new - mean_logp_old)
        w_k = torch.exp(mean_logp_new - mean_logp_old)  # (BG,)

        # Rank-level advantage A_k: (BG,)
        A_k = advantages[:, k - 1]  # (BG,)

        # Only keep valid BG where this rank exists
        w_k = w_k[valid_bg_mask]
        A_k = A_k[valid_bg_mask]

        # PPO-style clipped objective
        clipped_w_k = torch.clamp(w_k, 1.0 - clip_eps, 1.0 + clip_eps)
        obj_unclipped = w_k * A_k
        obj_clipped = clipped_w_k * A_k

        obj = torch.minimum(obj_unclipped, obj_clipped)  # (num_valid_bg,)
        total_loss_terms.append(obj)

    if not total_loss_terms:
        return torch.tensor(0.0, device=device, requires_grad=True)

    all_objs = torch.cat(total_loss_terms, dim=0)
    loss = -all_objs.mean()  # negative sign: we minimize loss, maximize objective

    return loss


# =========================
#  Synthetic "real-ish" CRS data
# =========================

def build_crs_examples() -> List[dict]:
    """
    Build a few conversational recommendation examples with:
      - prompt
      - good candidate list (high-relevance items)
      - bad candidate list (low-relevance items)
      - groundtruth relevant items

    We will later encode these with GPT-2 and compute per-rank rewards.
    """
    examples = []

    # Example 1: Sci-fi / Nolan-ish
    examples.append({
        "prompt": "User: I like Inception and Interstellar. Can you recommend some movies?\nAssistant:",
        "good_items": [
            "The Dark Knight",
            "Memento",
            "Tenet",
            "The Prestige",
        ],
        "bad_items": [
            "Frozen",
            "Cars",
            "Toy Story 3",
            "Moana",
        ],
        "relevant": {
            "The Dark Knight",
            "Memento",
            "Tenet",
            "The Prestige",
        },
    })

    # Example 2: Animated / Pixar-ish
    examples.append({
        "prompt": "User: I love Pixar movies like Up and Inside Out. What else should I watch?\nAssistant:",
        "good_items": [
            "Coco",
            "Finding Nemo",
            "Ratatouille",
            "Toy Story 3",
        ],
        "bad_items": [
            "The Godfather",
            "Fight Club",
            "Pulp Fiction",
            "The Shawshank Redemption",
        ],
        "relevant": {
            "Coco",
            "Finding Nemo",
            "Ratatouille",
            "Toy Story 3",
        },
    })

    return examples


def build_batch_from_examples(
    tokenizer,
    examples: List[dict],
    device: torch.device,
    num_ranks: int = 4,
    group_size: int = 2,
) -> Tuple[torch.Tensor, ...]:
    """
    Build a batch (B=len(examples), G=group_size, T=max_len) consisting of:
      - input_ids (prompt + item texts)
      - attention_mask
      - labels (ignore prompt tokens for RL with -100)
      - rank_ids (0 for prompt, 1..num_ranks for items)
      - rewards (per-rank rel_k in [0, 1])

    We use '|' as delimiter between items, so we know rank boundaries.
    """
    B = len(examples)
    G = group_size
    assert G == 2, "This demo assumes exactly 2 candidates per prompt (good/bad)."

    # Collect encoded sequences and metadata
    encoded_seqs = []  # list of (ids, prompt_len, ranks_info)
    # ranks_info: list[str] of length num_ranks with the item text at each position

    for ex in examples:
        prompt = ex["prompt"]
        good_items = ex["good_items"]
        bad_items = ex["bad_items"]
        relevant = ex["relevant"]  # set of relevant item strings

        # Encode prompt alone to know its token length
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Build candidate sequences as: prompt + " " + item1 + " | " + item2 + " | " ...
        # Ensure we have exactly num_ranks items per list.
        assert len(good_items) >= num_ranks
        assert len(bad_items) >= num_ranks

        def build_candidate_items(items: List[str]) -> Tuple[List[int], List[str]]:
            # Return token_ids and the item list (first num_ranks items)
            item_list = items[:num_ranks]
            # Join into a single string: " item1 | item2 | item3 | item4"
            candidate_text = " " + " | ".join(item_list)
            candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
            return candidate_ids, item_list

        # Good candidate
        good_ids, good_item_list = build_candidate_items(good_items)
        seq_good = prompt_ids + good_ids

        # Bad candidate
        bad_ids, bad_item_list = build_candidate_items(bad_items)
        seq_bad = prompt_ids + bad_ids

        encoded_seqs.append([
            (seq_good, prompt_len, good_item_list, relevant),
            (seq_bad, prompt_len, bad_item_list, relevant),
        ])

    # Find max length for padding
    max_len = max(len(seq) for ex in encoded_seqs for (seq, _, _, _) in ex)

    # Allocate tensors
    input_ids = torch.full((B, G, max_len), tokenizer.eos_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, G, max_len), dtype=torch.long, device=device)
    labels = torch.full((B, G, max_len), -100, dtype=torch.long, device=device)
    rank_ids = torch.zeros((B, G, max_len), dtype=torch.long, device=device)
    rewards = torch.zeros((B, G, num_ranks), dtype=torch.float, device=device)

    # Helper: find token id for '|' delimiter
    pipe_id_list = tokenizer.encode("|", add_special_tokens=False)
    if len(pipe_id_list) != 1:
        raise ValueError("Expected '|' to map to a single token in GPT-2 tokenizer.")
    pipe_id = pipe_id_list[0]

    # Fill tensors
    for b, ex in enumerate(encoded_seqs):
        for g, (seq_ids, prompt_len, item_list, relevant) in enumerate(ex):
            seq_len = len(seq_ids)
            input_ids[b, g, :seq_len] = torch.tensor(seq_ids, device=device)
            attention_mask[b, g, :seq_len] = 1

            # labels: ignore prompt tokens for RL with -100, train on candidate tokens
            labels[b, g, prompt_len:seq_len] = input_ids[b, g, prompt_len:seq_len]

            # rank_ids: from prompt_len onward, assign ranks based on '|' delimiters
            # algorithm: start at rank=1 after prompt; each time we see '|', we increment rank.
            current_rank = 1
            for t in range(prompt_len, seq_len):
                token_id = int(input_ids[b, g, t].item())
                if token_id == pipe_id:
                    # delimiter itself has no rank
                    current_rank += 1
                    if current_rank > num_ranks:
                        break
                else:
                    if 1 <= current_rank <= num_ranks:
                        rank_ids[b, g, t] = current_rank

            # rewards: rel_k in {0,1} based on whether item at rank k is in relevant set
            for k in range(num_ranks):
                item_k = item_list[k]
                rel_k = 1.0 if item_k in relevant else 0.0
                rewards[b, g, k] = rel_k

    return input_ids, attention_mask, labels, rank_ids, rewards


def compute_logprobs_old(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-wise logprobs under the "old" policy.

    In a real setup, this should be a frozen copy of the policy (old_model).
    For this demo, we simply reuse the same model and detach the logprobs.

    Returns:
      logprobs_old: (B, G, T)
    """
    B, G, T = input_ids.shape
    BG = B * G

    input_ids_flat = input_ids.view(BG, T)
    attention_mask_flat = attention_mask.view(BG, T)
    labels_flat = labels.view(BG, T)

    with torch.no_grad():
        log_probs_flat = compute_token_logprobs(model, input_ids_flat, attention_mask_flat)

    # Gather token logprobs for labels
    label_mask = labels_flat.ne(-100)
    safe_labels = labels_flat.clone()
    safe_labels[~label_mask] = 0
    token_logp_old = log_probs_flat.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp_old = token_logp_old * label_mask

    return token_logp_old.view(B, G, T)


# =========================
#  Main demo
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    model_name = "gpt2"  # you can swap to "gpt2-medium" etc. if you want
    num_ranks = 4
    group_size = 2  # good + bad per example
    num_steps = 3
    lr = 1e-5

    # Seeds
    torch.manual_seed(42)
    random.seed(42)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 has no padding token by default; assign eos_token as pad_token for convenience
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Build "real-ish" CRS examples
    examples = build_crs_examples()

    for step in range(num_steps):
        # 1) Build batch (B=len(examples), G=2)
        input_ids, attention_mask, labels, rank_ids, rewards = build_batch_from_examples(
            tokenizer=tokenizer,
            examples=examples,
            device=device,
            num_ranks=num_ranks,
            group_size=group_size,
        )

        # 2) Compute old logprobs
        logprobs_old = compute_logprobs_old(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # 3) Compute Rank-GRPO loss
        loss = rank_grpo_loss(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            logprobs_old=logprobs_old,
            rank_ids=rank_ids,
            rewards=rewards,
            clip_eps=0.06,
        )

        # 4) Backprop + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step+1}/{num_steps} - Rank-GRPO loss: {loss.item():.4f}")

    print("Demo finished. You can now adapt this code to your own CRS setup.")


if __name__ == "__main__":
    main()
