# 🧠 Qwen 3

### Third-Generation Reasoning-Centric **Text** LLMs by Alibaba Group (2025)

> Scope note: **Qwen 3 is a text LLM family** (dense + MoE, 0.6B–235B). Vision/Audio models are separate lines and not part of this technical report.

---

## 🚀 Overview

Qwen 3 introduces a unified pretraining → post-training recipe emphasizing **high‑quality synthetic reasoning**, **long‑context training**, and **controllable thinking**. It also provides an efficient **strong‑to‑weak distillation** route for small models.

---

## 🏗️ Model Architecture (at a glance)

- **Backbone:** Decoder‑only Transformer (dense & MoE variants)
- **Tokenizer:** ~151K vocabulary
- **Context Length:** 32K base, extendable to **128K** via long‑context training (YaRN)
- **Attention:** Grouped Query Attention (GQA)
- **Norm/Act:** RMSNorm, SwiGLU
- **Training precision:** bfloat16, ZeRO‑3

---

## 🧩 Pretraining

- **Objective:** Causal LM (next‑token prediction)
- **Scale:** ~**18T tokens**
- **Mixture:** filtered web/books/conversations; **code**; **math/science**; **multilingual** (>40 languages); **synthetic reasoning** (CoT)
- **Data QA:** LLM‑based quality filtering, dedup, benchmark decontamination
- **Curriculum:** 3 stages with growing max sequence length and reasoning intensity
  1. ~6T @ 4K → core language
  2. ~8T @ 8K–16K → reasoning + context expansion
  3. ~4T @ 32K → long‑context + code

**Long‑context method:** **YaRN** (RoPE enlargement) + training on concatenated long documents; objective remains NTP.

---

## 🎓 Post‑Training Pipeline (4 stages)

1. **Long‑CoT Cold Start**  
   Large‑scale synthetic CoT traces with verified answers to seed structured reasoning.

2. **Reasoning RL**  
   Verifiable‑reward RL (math/coding/logic) with stability tricks (e.g., temperature annealing, reward normalization).

3. **Thinking‑Mode Fusion**  
   Unite _thinking_ and _non‑thinking_ behaviors in one chat model.

   - Prompt/API switches (e.g., `/think`, `/no think`, optional `<think>…</think>` blocks).
   - Fusion SFT set = **thinking** samples (via rejection sampling from the reasoning model) + **non‑thinking** samples (instruction, coding, multilingual, translation, role‑play, QA).
   - Improves instruction adherence and enables robust **mode switching**.

4. **General RL**  
   Preference‑based tuning (hybrid DPO‑RLHF) for instruction following, tone, safety, and factuality while preserving reasoning.

---

## ⏱️ Thinking Budget (inference‑time control)

Allocate a **budget of “thinking tokens”** per request. Higher budgets → stronger reasoning (more steps); lower budgets → lower latency. This leverages the fused mode interface to make reasoning depth **user‑controllable**.

---

## 🧬 Strong‑to‑Weak Distillation (order clarified)

**Goal:** Train smaller **students** (0.6B–8B) efficiently under large **teacher** models without re‑running the full RL pipeline.

**Two‑phase order (correct):**

1. **Off‑policy distillation (first):**

   - Train the student on **fixed teacher outputs/logits** (combined thinking + non‑thinking responses).
   - Transfers broad competence and formatting quickly.

2. **On‑policy distillation (second):**
   - Let the **student** roll out on prompts; **align** its behavior to the teacher’s via KL/logit matching on the student’s own trajectories.
   - Reduces distribution mismatch and sharpens reasoning.

**Why this path:** Comparable or better quality than RL with **~10× less compute**, producing small models that retain the teacher’s reasoning and alignment behavior.

---

## ⚙️ Key Takeaways

- **Text‑only LLM family:** Qwen 3 focuses on language models; multimodal lines are separate.
- **Thinking‑Mode Fusion + Budget:** Practical, controllable reasoning depth per query.
- **Efficient scaling:** Strong‑to‑weak distillation (off‑policy → on‑policy) yields capable small models.
- **Data quality > data volume:** Heavy use of filtered and synthetic reasoning data.
- **Curriculum + YaRN:** Smooth extension to 128K context for multi‑document tasks.

---

## 📚 Citation

Qwen Team. “**Qwen3 Technical Report**” arXiv:2505.09388 (2025).
