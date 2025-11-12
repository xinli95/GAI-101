---
layout: default
title: Reinforcement Learning from Human Feedback
description: Comprehensive guide with PPO, DPO, GRPO, and RLVR notes.
permalink: /rlhf.html
---

# Reinforcement Learning from Human Feedback (RLHF) — A Comprehensive Guide

---

## 1. Introduction

Large Language Models (LLMs) are trained in three main stages:

1. **Pretraining** – predict the next token from large text corpora.
2. **Supervised Fine‑Tuning (SFT)** – fine‑tune on curated prompt–response pairs.
3. **Reinforcement Learning from Human Feedback (RLHF)** – align model outputs with human preferences.

---

## 2. The Core Idea

Instead of maximizing likelihood of text, RLHF rewards the model for producing _preferred_ outputs.

$$
\max_{\theta} \; \mathbb{E}_{x,y\sim\pi_\theta}\left[r(x,y) - \beta\, KL(\pi_\theta(y|x) \,\|\, \pi_{ref}(y|x))\right]
$$

- **r(x, y)** – reward from human feedback or verifier.
- **KL term** – keeps policy close to reference model.
- **β** – strength of regularization.

---

## 3. Key Components

| Component             | Role                           |
| --------------------- | ------------------------------ |
| **Policy model**      | The LLM being tuned.           |
| **Reference model**   | Frozen base model (stability). |
| **Reward model (RM)** | Scores outputs by preference.  |
| **RL algorithm**      | Updates policy from rewards.   |

---

## 4. PPO — Proximal Policy Optimization

**PPO** is the traditional RLHF algorithm (used by OpenAI 2019).

### 🧠 Intuition

- Sample responses from the current policy.
- Evaluate them with a reward model.
- Estimate _advantages_ using a critic (value head).
- Update the model with a **clipped policy objective**.

### 📘 Objective

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\left[\min\big(r_t A_t,\; \text{clip}(r_t, 1\!\pm\!\epsilon) A_t\big)\right]
$$

where $r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$ and $A_t$ is the advantage (via **GAE**).

**Advantage intuition.** The advantage quantifies “how much better than average” an action is in a given state: $A_t = Q_\pi(s_t, a_t) - V_\pi(s_t)$. Positive advantages push the policy toward the sampled action, while negative values suppress it.

### Generalized Advantage Estimation (GAE)

GAE smooths the noisy Monte Carlo estimate of $A_t$ by blending multi-step temporal-difference (TD) errors:

$$
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

The decay term $(\gamma \lambda)^l$ trades off bias vs. variance: $\lambda \to 1$ approaches Monte Carlo returns, while smaller $\lambda$ relies more on the critic’s bootstrapped values. In practice, PPO computes $\delta_t$ with the critic head, runs the reverse-time sum above, and normalizes $\hat{A}_t$ before plugging it into the clipped loss.

### Critic vs. Reward Model

- **Reward model** provides per-sample scalar scores derived from human preference data or verifiable checks. It approximates the _external_ reward that PPO tries to maximize.
- **Critic (value head)** estimates $V_\phi(s)$, the expected _future_ reward from a partial generation. It is trained with a regression loss to match Monte Carlo returns and is only used to produce low-variance advantages (GAE).

In short, the reward model shapes _what_ the policy wants, while the critic stabilizes _how_ gradients propagate through PPO updates.

### ✅ Pros / ❌ Cons

| Pros                                        | Cons                          |
| ------------------------------------------- | ----------------------------- |
| Stable updates, per‑token credit assignment | Heavy (critic + reward model) |
| KL regularization controls drift            | Many hyperparameters          |

---

## 5. DPO — Direct Preference Optimization

**DPO** (Rafailov et al., 2023) removes the reward and critic entirely.

### 🧩 Idea

Train directly from preference pairs $(x, y^+, y^-)$: the model should make $y^+$ more likely than $y^-$ while staying close to the base model.

### 📘 Loss

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\Big( \beta\big[(\log\pi_\theta(y^+\mid x) - \log\pi_{ref}(y^+\mid x)) - (\log\pi_\theta(y^-\mid x) - \log\pi_{ref}(y^-\mid x))\big] \Big)
$$

### ✅ Pros / ❌ Cons

| Pros                  | Cons                               |
| --------------------- | ---------------------------------- |
| No RL loop, no critic | Offline (needs preference dataset) |
| Simple, stable        | No dynamic reward feedback         |

---

## 6. GRPO — Group Relative Policy Optimization

**GRPO** is PPO without a critic. Instead of a learned value head, it normalizes rewards within a _group_ of responses per prompt.

### ⚙️ Process

1. For each prompt, sample multiple responses.
2. Score each with reward model or verifier.
3. Compute relative advantages:  
   $A_i = (r_i - \mu_G)/\sigma_G$
4. Apply PPO‑style update using these group‑normalized advantages.

### ✅ Pros / ❌ Cons

| Pros                               | Cons                                   |
| ---------------------------------- | -------------------------------------- |
| No value head, cheaper training    | Requires multiple responses per prompt |
| Works well with verifiable rewards | Slightly higher generation cost        |

---

## 7. RLVR — Reinforcement Learning with Verifiable Rewards

**RLVR** defines a reward via objective verification instead of a learned model.

### 💡 Idea

If correctness can be _checked_ (math, code, logic), we can replace the reward model with an automatic **verifier**.

### ⚙️ Steps

1. Generate response(s) for prompt.
2. Run a verifier or test suite → numeric reward.
3. Optimize policy with PPO or GRPO.

### ✅ Pros / ❌ Cons

| Pros                         | Cons                                  |
| ---------------------------- | ------------------------------------- |
| Objective, bias‑free rewards | Only works for verifiable tasks       |
| Pairs naturally with GRPO    | Needs reliable checker infrastructure |

---

## 8. Comparison Summary

| Method   | Reward Source           | Critic?  | On‑policy? | Multi‑response? | Typical Use             |
| -------- | ----------------------- | -------- | ---------- | --------------- | ----------------------- |
| **PPO**  | Learned reward model    | ✅       | ✅         | ❌              | Classic RLHF            |
| **DPO**  | Human preference pairs  | ❌       | ❌         | ❌              | Offline alignment       |
| **GRPO** | Reward model / verifier | ❌       | ✅         | ✅              | Efficient online tuning |
| **RLVR** | Verifiable function     | Optional | ✅         | ✅              | Code/math correctness   |

**Shared advantage theme.** PPO leans on critic-based GAE, GRPO normalizes sibling rewards into a $z$-score, and RLVR can adopt either strategy (critic or group). Even DPO, despite being offline, uses the log-prob gap between preferred and rejected responses as a surrogate “advantage.” Each objective therefore scales the policy update by a relative better-vs-worse signal, keeping the optimization logic consistent across methods.

---

## 9. Conceptual Links

All methods aim to maximize expected reward while constraining divergence from a base model:

$$
\max_\pi \; \mathbb{E}[r(x,y)] - \beta \, KL(\pi \,\|\, \pi_{ref})
$$

They differ mainly in _how_ reward and advantage are estimated:

| Method   | Reward Estimation | Advantage Estimation |
| -------- | ----------------- | -------------------- |
| **PPO**  | Learned RM        | Critic (GAE)         |
| **DPO**  | Direct preference | Log‑prob difference  |
| **GRPO** | RM or verifier    | Group z‑score        |
| **RLVR** | Verifier          | Group or critic      |

---

## 10. Modern RLHF Stack

1. **SFT** — instruction fine‑tuning.
2. **DPO / ORPO** — offline preference alignment.
3. **GRPO / RLVR** — online verifiable reinforcement.
4. **Self‑rewarding / distillation** — model refines itself with reasoning/verifier feedback.

---

## 11. Key Takeaways

- **RLHF** aligns models using rewards, not just likelihood.
- **PPO** is the classic but heavy approach.
- **DPO** removes RL complexity using a closed‑form objective.
- **GRPO** simplifies PPO by replacing the critic with group normalization.
- **RLVR** grounds RLHF in _verifiable correctness_ rather than human scoring.

Together, they form a spectrum — from full RL to lightweight preference optimization — that powers modern aligned LLMs.

---
