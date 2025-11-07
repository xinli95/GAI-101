# 🧠 Phi-4 (14B)
### High-Quality, Reasoning-Centric Language Model

This document summarizes the **training recipe** for **Phi-4 (14B)** — Microsoft’s reasoning-optimized large language model.  
Phi-4 builds on the Phi family’s design philosophy: **quality over quantity**, emphasizing synthetic data curation, curriculum learning, and long-context adaptation.

---

## 🏗️ Model Architecture
- **Type:** Decoder-only Transformer, 14 billion parameters  
- **Tokenizer:** tiktoken (100 K vocabulary)  
- **Context length:** 4 K → extended to **16 K** during mid-training  
- **Backbone:** Derived from Phi-3-medium with full attention (no sliding window)  
- **Training horizon:** ≈ **10 trillion tokens**

---

## 🧩 1. Pre-Training

### **Objective**
Develop a reasoning-oriented model with strong math, coding, and logic capabilities using a **hybrid synthetic + filtered organic** data mixture.

### **Data Composition**
- **Total tokens:** ≈ **10 trillion**
- **Synthetic data (~40%)** generated through multiple workflows:
  - Seed curation from educational, scientific, and coding sources.
  - Iterative self-revision pipelines to improve correctness and style.
  - Task inversion to generate instructions from solutions.
  - Automated validation of code and math expressions.
- **Organic data (~60%)** includes filtered web, books, and code with high linguistic diversity.
  - Multi-stage LLM filtering (≈ 1M labels).
  - HTML/PDF/TeX extraction for equations and tables.
  - Decontamination against public benchmarks (7–13-gram match).

| Data Type | Fraction | Unique Tokens | Approx. Epochs |
|------------|-----------|----------------|----------------|
| Filtered Web | 15 % | 1.3 T | 1.2 × |
| Web Rewrites (Synthetic) | 15 % | 0.29 T | 5.2 × |
| Synthetic Reasoning / Instruction | 40 % | 0.29 T | 13.8 × |
| Code (raw + synthetic) | 20 % | 0.82 T | 2.4 × |
| Academic / Books | 10 % | 0.58 T | 1.7 × |

> Total effective tokens ≈ **10 T**, with heavier emphasis on synthetic repetition.

### **Optimization**
- Learning rate = 3e-4 (cosine decay with warm-up)
- Weight decay = 0.1  
- Global batch size ≈ 5.8K tokens × GPU shards  
- Stable optimization tuned via short-horizon pilot runs

---

## 🧩 2. Mid-Training (Long-Context Adaptation)

### **Goal**
Expand context window (4 K → 16 K) and strengthen multi-document reasoning.

### **Implementation**
- Continue training from the pre-trained checkpoint for ≈ **250 B tokens**
- **30 %** long-context samples (> 8 K tokens), mostly from academic and code data
- **70 %** reused recall data from pre-training
- Extended synthetic data via concatenation and padding
- **RoPE base frequency = 250 K**
- Learning rate reduced by 10×

---

## 🧩 3. Post-Training

Transform Phi-4 into a **safe, instruction-following assistant** through three stages.

### **Stage 1 – Supervised Fine-Tuning (SFT)**
- **Tokens:** ≈ 8 billion (ChatML format)
- **Learning rate:** 1e-6
- **Domains:** math, coding, reasoning, safety, multilingual (≈ 40 languages)
- **Goal:** teach structured instruction-following and safe refusal.

### **Stage 2 – Direct Preference Optimization (DPO)**
Two rounds applied to the SFT model:

1. **Pivotal Token DPO (≈ 300 K pairs)**  
   - Targets token-level corrections for reasoning and code tasks.  
   - Pairs identified using pivotal token search.

2. **Judge-Guided DPO (≈ 850 K pairs)**  
   - GPT-4o-based judges label positive vs. negative responses.  
   - Scoring dimensions: accuracy, completeness, style.

### **Stage 3 – Hallucination Mitigation**
- Trains explicit **refusal behavior** for low-confidence or nonsensical inputs.
- Synthetic “bogus question” datasets encourage safe decline responses.
- Achieves ~80% reduction in hallucinations on SimpleQA tasks.

---

## 📊 Scale Summary

| Stage | Token Count / Samples | Purpose |
|--------|----------------------|----------|
| **Pre-training** | ≈ 10 T tokens (4 K ctx) | Foundational reasoning |
| **Mid-training** | ≈ 0.25 T tokens (16 K ctx) | Long-context adaptation |
| **SFT** | ≈ 8 B tokens | Instruction & safety |
| **DPO (2 rounds)** | ≈ 1.1 M pairs | Preference alignment & refinement |

---

## ⚙️ Key Insights

- **Synthetic data dominates:** high-quality synthetic reasoning drives the model’s logic performance.  
- **Curriculum learning:** data mixture gradually transitions from natural web → structured reasoning → supervised refinement.  
- **Mid-training as context curriculum:** progressively longer samples improve long-context stability.  
- **Token-level alignment:** pivotal-token DPO adds fine-grained reward signal.  
- **Refusal training:** improves safety and factual restraint.  
- **Total data exposure:** ≈ **10.3 trillion tokens** (pre + mid + post).

---

## 📚 Citation

> Microsoft Research.  
> **Phi-4 Technical Report: Scaling Small Language Models with High-Quality Synthetic Data.**  
> *arXiv preprint arXiv:2412.08905 (2024).*
