# 🧠 Phi-4-Mini-Reasoning

### Exploring the Limits of Small Reasoning Language Models in Math

This repository summarizes the **training recipe** used to develop **Phi-4-Mini-Reasoning (3.8 B)** — a small yet powerful reasoning model derived from Microsoft’s Phi-4-Mini. The approach demonstrates that with deliberate data design and multi-stage training, compact models can achieve reasoning performance comparable to or surpassing much larger models.

---

## 🚀 Overview

**Goal:** Enhance formal reasoning in small language models (SLMs) through a **four-stage continual training recipe** combining large-scale distillation, supervised refinement, preference learning, and reinforcement learning with verifiable rewards.

**Model:** [Phi-4-Mini (3.8 B)](https://arxiv.org/abs/2503.01743)  
**Base Domain:** Mathematical reasoning  
**Final Model:** _Phi-4-Mini-Reasoning_

---

## 🧩 Training Recipe

### **Stage 1 — Large-Scale Mid-Training (Distillation)**

- **Purpose:** Inject general Chain-of-Thought (CoT) reasoning capabilities into the base model.
- **Data:** ~1.6 M unique math problems (≈ 10 M reasoning rollouts) generated and verified from LLMs such as DeepSeek-R1 (671 B).
- **Process:**
  - Train with next-token prediction on verified CoT answers.
  - Apply rejection sampling to keep only correct responses.
  - Use **sequence packing** for efficiency.
- **Key Parameters:**
  - Sequence length = 16 K
  - Batch size = 128
  - Learning rate = 1e-5
  - 5 epochs with 10% warmup

### **Stage 2 — Supervised Fine-Tuning (SFT)**

- **Purpose:** Refine reasoning quality and generalization on more difficult, high-quality data.
- **Data:** A curated subset of mid-training data focusing on challenging, diverse math domains (≥ college level).
- **Process:**
  - Train without sequence packing to allow the model to learn when to stop generating.
- **Key Parameters:**
  - Sequence length = 20 K
  - Batch size = 128
  - Learning rate = 1e-5
  - 5 epochs

### **Stage 3 — Rollout Preference Learning (DPO)**

- **Purpose:** Leverage both correct and incorrect rollouts to align model preferences toward verified reasoning paths.
- **Data:** Preference pairs built from accepted (correct) and rejected (incorrect) DeepSeek-R1 outputs.
- **Process:**
  - Use Direct Preference Optimization (DPO) with the Stage 2 model as reference.
  - Single-epoch training with long sequence contexts.
- **Key Parameters:**
  - Sequence length = 16 K
  - Learning rate = 5e-7
  - 1 epoch

### **Stage 4 — Reinforcement Learning with Verifiable Reward (RLVR)**

- **Purpose:** Further improve reasoning accuracy through online RL guided by verifiable correctness signals.
- **Reward Definition:** +1 for verified correct final answers; −1 otherwise.
- **Algorithms:** Proximal Policy Optimization (PPO) and Group-based Relative Policy Optimization (GRPO).
- **Stability Enhancements:**
  - **Prompt Optimization:** Keep prompts producing uniform-length responses.
  - **Reward Re-balancing:** Oversample and balance positive/negative samples; filter overly easy prompts.
  - **Temperature Annealing:** Linearly decay sampling temperature from 1.0 → 0.6 during early training, then fix at 0.6.
- **Key Parameters:**
  - Sequence length = 25 K
  - Learning rate = 5e-7

---

## 📊 Performance

| Model                 | AIME 24  | MATH-500 | GPQA Diamond |
| --------------------- | -------- | -------- | ------------ |
| **Phi-4-Mini (base)** | 10.0     | 71.8     | 36.9         |
| + Mid-Training        | 30.0     | 82.9     | 42.6         |
| + Fine-Tuning         | 43.3     | 89.3     | 48.3         |
| + Rollout DPO         | 50.0     | 93.6     | 49.0         |
| **+ RLVR (final)**    | **57.5** | **94.6** | **52.0**     |

> Phi-4-Mini-Reasoning (3.8 B) outperforms larger reasoning models such as  
> DeepSeek-R1-Distill-Qwen-7B (+3.2 pts on MATH-500) and  
> DeepSeek-R1-Distill-Llama-8B (+7.7 pts on MATH-500).

---

## ⚙️ Training Summary

| Stage            | Objective                | Data Scale          | Sequence Len | LR   | Epochs |
| ---------------- | ------------------------ | ------------------- | ------------ | ---- | ------ |
| 1 – Mid-Training | CoT distillation (LM)    | ~1.6 M samples      | 16 K         | 1e-5 | 5      |
| 2 – Fine-Tuning  | High-quality SFT         | Selected subset     | 20 K         | 1e-5 | 5      |
| 3 – Rollout DPO  | Preference alignment     | Paired rollouts     | 16 K         | 5e-7 | 1      |
| 4 – RLVR         | Verifiable RL (GRPO/PPO) | Verified math tasks | 25 K         | 5e-7 | —      |

---

## 🧠 Key Insights

- **Scale ≠ Reasoning:** Carefully designed data and stage-wise training can unlock strong reasoning in small models.
- **Quality & Verification Matter:** Verified synthetic CoT trajectories form the foundation of success.
- **RL Stability is Crucial:** Prompt filtering, balanced rewards, and temperature scheduling enable stable improvement.

---
