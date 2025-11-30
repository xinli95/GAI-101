# 📘 Rank-GRPO: A Step-by-Step Tutorial for Conversational Recommendation RL

## Table of Contents

1. Overview
2. Why GRPO Fails for Ranking
3. Rank-GRPO: Core Ideas
4. Understanding Rank-Level Rewards
5. Rank-Level Importance Ratios
6. Rank-Level Advantages
7. Rank-Level PPO Objective
8. Full Algorithm (Annotated)
9. Code Skeleton
10. Putting It All Together
11. Practical Tips
12. Citation

---

# 1. Overview

Rank-GRPO (_Rank-based Group Relative Policy Optimization_) is a reinforcement learning algorithm designed to train LLM-based **conversational recommender systems (CRS)**.

It fixes key failures of GRPO/PPO when used for _ranked_ recommendation lists, such as:

```
1. The Dark Knight
2. Memento
3. Tenet
4. The Prestige
```

Unlike normal text generation, recommendation outputs have **structure**:

- Each rank (1 → N) is a separate decision.
- Top ranks matter more.
- Rewards must be **rank-indexed**, **causal**, and **local**.

Rank-GRPO aligns **reward**, **credit assignment**, and **policy update** at the **rank level**.

---

# 2. Why GRPO Fails for Ranking

## Problem 1 — Non-causal credit assignment

GRPO often uses a single sequence-level reward such as:

$$
DCG@20 = \sum_{j=1}^{20} \frac{rel_j}{\log_2(j+1)}
$$

This reward is applied **uniformly** to **every token** in the list — even tokens in poorly ranked tail items benefit from the success of top-ranked items.

This destroys ranking quality.

---

## Problem 2 — Token-level importance ratios are misaligned

GRPO computes token-wise importance ratios:

$$
w_t = \frac{\pi_{\text{new}}(y_t)}{\pi_{\text{old}}(y_t)}
$$

But ranking choices are made at the **item** level, not token level.  
Because items have variable lengths, token-level ratios cause:

- length bias
- unstable updates
- incorrect credit assignment

---

# 3. Rank-GRPO: Core Ideas

Rank-GRPO restructures GRPO around **rank positions**.

| Concept            | GRPO     | Rank-GRPO                  |
| ------------------ | -------- | -------------------------- |
| Reward unit        | sequence | **rank**                   |
| Advantage          | sequence | **rank**                   |
| Importance ratio   | token    | **rank**                   |
| Update granularity | token    | **tokens grouped by rank** |

Everything becomes “rank-aware.”

---

# 4. Understanding Rank-Level Rewards

For a generated ranked list:

$$
y = \big(y^{(1)}, y^{(2)}, \ldots, y^{(N)}\big)
$$

Rank-GRPO needs a reward for each rank $k$.

Two options:

---

## Option A — Causal DCG (DCG@k:N)

$$
r_k = \sum_{j=k}^{N} \frac{rel_j}{\log_2(j+1)}
$$

- Rank 1 gets full DCG.
- Rank 2 ignores rank 1’s relevance.
- Rank $N$ gets only its own relevance.

This fixes reward leakage from earlier ranks.

---

## Option B — Simple relevance ($rel_k$) (EXP$\infty$ variant)

$$
r_k = rel_k
$$

Where $rel_k \in \{0,1\}$ indicates whether the item at rank $k$ is relevant.

This variant performs best in practice and is easiest to implement.

---

# 5. Rank-Level Importance Ratios

Each item $y^{(k)}$ consists of many tokens.

Let $\text{tokens}(k)$ be the set of tokens belonging to rank $k$.

### Step 1 — Mean log-probability per rank

$$
\bar{\log p}_{\text{new},k}
= \frac{1}{\lvert \text{tokens}(k) \rvert}
\sum_{t \in \text{tokens}(k)}
\log \pi_{\text{new}}(y_t)
$$

Similarly for $\pi_{\text{old}}$:

$$
\bar{\log p}_{\text{old},k}
= \frac{1}{\lvert \text{tokens}(k) \rvert}
\sum_{t \in \text{tokens}(k)}
\log \pi_{\text{old}}(y_t)
$$

---

### Step 2 — Rank-level importance ratio

$$
w_k = \exp\Big(
\bar{\log p}_{\text{new},k} - \bar{\log p}_{\text{old},k}
\Big)
$$

This is the **ratio of geometric mean probabilities** of all tokens in the item.

It correctly normalizes across different item lengths.

---

# 6. Rank-Level Advantages

Given rewards

$$
(r_1, r_2, \ldots, r_N),
$$

compute group-relative **rank advantages**:

$$
A_k = \frac{r_k - \text{mean}(r)}{\text{std}(r)}.
$$

This is the core “group-relative” idea from GRPO, applied per rank instead of per sequence.

---

# 7. Rank-Level PPO Objective

For each rank $k$:

$$
J_k = \min \left(
w_k A_k,\;
\text{clip}(w_k, 1 - \epsilon, 1 + \epsilon)\, A_k
\right).
$$

Combine all ranks:

$$
J = \frac{1}{N} \sum_{k=1}^N J_k,
$$

and define the loss as

$$
\mathcal{L} = -J.
$$

Because optimizers **minimize** loss, this is equivalent to maximizing the PPO-style objective $J$.

---

# 8. Full Algorithm (Annotated)

**For each training step:**

1. **Rollout**

   - For each prompt $x$, sample $G$ candidate ranked lists $y_i$.
   - For each candidate and each rank $k$, compute $r_{i,k}$ (e.g., $rel_k$ or DCG@k:N).

2. **Token–rank mapping**

   - For each candidate list, mark tokens that belong to rank $k$ (e.g., using delimiters like `"|"`).
   - Compute token log-probabilities under both the **old** and **current** policies.

3. **Rank-wise statistics**

   - For each rank $k$, compute $\bar{\log p}_{\text{new},k}$ and $\bar{\log p}_{\text{old},k}$,  
     then obtain $w_k$ and $A_k$ as above.

4. **Rank-level PPO objective**

   - Compute
     $$
     J_k = \min\big(w_k A_k,\; \text{clip}(w_k, 1-\epsilon, 1+\epsilon) A_k\big).
     $$
   - Accumulate $J_k$ over all ranks and all candidates.

5. **Loss and update**
   - Set $\mathcal{L} = -\text{mean}(J_k)$ across all samples and ranks.
   - Backpropagate through the current model and update its parameters.

---

# 9. Code Skeleton

Below is a minimal code-style sketch of Rank-GRPO (PyTorch-like):

```python
for k in range(1, N + 1):
    # mask tokens that belong to rank k
    mask = (rank_ids == k) & label_mask          # (BG, T)

    # count how many tokens we have for this rank, per sample
    token_counts = mask.sum(dim=-1)              # (BG,)

    # mean log-probabilities per rank (geometric mean in log-space)
    sum_new = (token_logp_new * mask).sum(dim=-1)
    sum_old = (token_logp_old * mask).sum(dim=-1)

    mean_new = sum_new / token_counts
    mean_old = sum_old / token_counts

    # rank-level importance ratio
    w_k = torch.exp(mean_new - mean_old)

    # rank-level advantage
    A_k = (rewards[:, k] - rewards.mean(dim=-1)) / (rewards.std(dim=-1) + eps)

    # PPO clipping at rank level
    w_clip = torch.clamp(w_k, 1.0 - eps_clip, 1.0 + eps_clip)
    obj_unclipped = w_k * A_k
    obj_clipped   = w_clip * A_k

    J_k = torch.minimum(obj_unclipped, obj_clipped)   # (BG,)

    all_J.append(J_k)

loss = -torch.mean(torch.cat(all_J, dim=0))
loss.backward()
optimizer.step()
```

This is the core Rank-GRPO logic: everything is aggregated by **rank**, not token.

---

# 10. Putting It All Together

Rank-GRPO delivers:

- **Rank-based rewards** → avoids non-causal leakage from other positions.
- **Rank-based importance sampling** → stable, length-normalized updates.
- **Rank-based credit assignment** → correct gradients for each decision point.
- **Rank-based PPO** → controlled policy updates and improved stability.

In practice, this lets relatively small LLMs (0.5B–3B) achieve or surpass the performance of much larger models (e.g., GPT-4-class) on conversational recommendation tasks, especially on **NDCG@20 / Recall@20**.

---

# 11. Practical Tips

- Use a **clear delimiter** (e.g., `"|"` or newline) to separate items so that you can reliably construct `rank_ids`.
- Start with **simple binary relevance** rewards ($rel_k \in \{0,1\}$) before moving to DCG-based variants.
- Use relatively small PPO clip values, e.g. $\epsilon \in [0.05, 0.1]$.
- Optionally add a **KL penalty** to keep the policy close to the SFT model.
- Carefully debug `rank_ids` alignment and reward computation — bugs there will silently break training.

---

# 12. Citation

If you use this method, please cite the original paper:

```
@article{gong2025rankgrpo,
  title   = {Rank-GRPO: Direct Preference Optimization for Conversational Recommendation},
  author  = {Zhitao Gong and Yifan Wang and Zihan Zhang and others},
  journal = {arXiv preprint arXiv:2511.11653},
  year    = {2025}
}
```

---
