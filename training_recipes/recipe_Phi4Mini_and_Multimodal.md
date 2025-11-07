# 🧠 Phi-4-Mini and Phi-4-Multimodal
### Compact Yet Powerful Language and Vision-Language Models

This repository summarizes the **training recipes** for **Phi-4-Mini (3.8 B)** and **Phi-4-Multimodal (5.6 B)**, introduced by Microsoft (2025).  
Both models aim to push the boundaries of compact model performance in reasoning, coding, and multimodal understanding through high-quality data and efficient architecture design.

---

## 🧩 Phi-4-Mini (3.8 B)

### 🔹 Architecture
- **Type:** Decoder-only Transformer (32 layers, hidden = 3072)
- **Tokenizer:** 200 K multilingual vocabulary
- **Context Length:** up to 128 K via **LongRoPE**
- **Attention:** **Group Query Attention** (24 query heads / 8 KV heads → ⅓ KV cache size)
- **Shared input/output embeddings** to improve efficiency and reduce memory footprint.

---

### 🏋️ Training Pipeline

#### **1. Pre-Training**
- **Objective:** Build strong general-purpose reasoning, language, and coding capability.  
- **Corpus:** ≈ **5 trillion tokens** of high-quality, filtered data.
- **Data Composition:**
  - Broader coverage of natural language, mathematics, and programming domains.
  - Carefully filtered for quality, safety, and factuality.
  - Augmented with synthetic instruction and reasoning data from Phi-4 pipeline.
- **Optimization:**
  - Learning-rate scaling: \( LR^* = B × D^{-0.32} \), tuned across 12.5 B–50 B tokens.
  - Sequence length during pre-training: **128 K tokens**.

#### **2. Post-Training (Instruction & Function Tuning)**
- **Goal:** Strengthen instruction following, summarization, and function-calling capabilities.
- **Data:**
  - Mixture of synthetic and curated real SFT data.
  - Includes code completion, long-context summarization, reasoning QA, and function execution tasks.
- **Training Approach:**
  - Full model fine-tuning (no adapters).
  - Instruction quality and factual grounding prioritized.

---

## 🖼️ Phi-4-Multimodal (5.6 B)

### 🔹 Core Design
- Built on top of the frozen **Phi-4-Mini** backbone.
- Extends Phi-4-Mini into a **unified multimodal model** via **Mixture-of-LoRAs**, enabling text, image, and speech inputs.
- Adds modality-specific **LoRA adapters** for visual and audio understanding without degrading the text model performance.

---

## 🧠 Mixture-of-LoRAs Framework
- Each modality (vision, audio, text) attaches its own lightweight LoRA adapter.
- The base LM remains frozen — only LoRA and modality-specific encoders are trained.
- Benefits:
  - Modular expansion to new modalities.
  - Efficient parameter utilization.
  - No interference between modalities.

---

## 🧩 Vision Modality

### **Architecture**
- **Encoder:** SigLIP-400 M, fine-tuned via LLM2CLIP.
- **Projector:** 2-layer MLP mapping vision embeddings → LM token space.
- **LoRA (V):** applied to all linear layers in the decoder (≈ 370 M trainable parameters).

### **Training Stages**
1. **Projector Alignment:** Train projector only on captioning datasets.
2. **Joint Vision Pre-Training:** Train both projector and encoder on ~0.5 T tokens of interleaved image-text data.
3. **Generative Vision-Language SFT:** Fine-tune LoRA, encoder, and projector on curated multimodal instruction datasets.
4. **Multi-Frame Training:** Extend to 64 K context for temporal and multi-image reasoning.

### **Data & Configs**
- Image resolution up to **1344 × 1344**.
- Dynamic multi-crop sampling used for scale invariance.
- Training data mixture includes captioning, VQA, OCR, and grounded reasoning tasks.

---

## 🔊 Speech / Audio Modality

### **Architecture**
- **Input:** 80-dim log-Mel filter-bank (10 ms frames).
- **Encoder:** 3 conv layers + 24 conformer blocks (1024 attention dim).
- **Projector:** 2-layer MLP (1024 → 3072).
- **LoRA (A):** applied to all attention and MLP layers (rank = 320) → ≈ 460 M parameters.

### **Training Stages**
1. **Pre-Training:**
   - Align audio encoder to LM semantic space.
   - Data: ≈ **2 million hours** of speech-text pairs (8 languages).
   - LM frozen; projector trained for 50 K steps (LR = 4e-5).
2. **Post-Training:**
   - Unlock multimodal instruction-following ability.
   - Data: ≈ **100 million weighted SFT samples** across ASR, AST, SQA, audio summarization, and comprehension tasks.
   - LM frozen; train LoRA and projector for 50 K steps (LR = 1e-4).
   - Max audio input length: **30 minutes (22.5 K tokens)**; supports inference up to 2.8 hours with 128 K context.

---

## 🗣️ Vision-Speech Joint Training
- Conducted after independent vision and speech post-training.
- **Frozen:** base LM, audio encoder, audio projector.
- **Trainable:** vision encoder, projector, and vision LoRA.
- Dataset: Combined **vision-speech SFT data** plus interleaved language-only and vision-language samples.

---

## 📊 Data Overview

| Modality | Stage | Data Scale | Notes |
|-----------|--------|------------|-------|
| **Language** | Pre-train | ~5 T tokens | Curated web + synthetic math/coding data |
| | Post-train | — | Instruction following, summarization, function calling |
| **Vision** | Pre-train | 0.5 T tokens | Interleaved image-text corpus |
| | SFT | 0.3 T tokens | Multimodal instruction fine-tuning |
| **Speech/Audio** | Pre-train | 2 M hours | Multilingual speech-text pairs |
| | Post-train | 100 M samples | ASR, AST, QA, summarization, and comprehension |

---

## ⚙️ Key Takeaways

- **Modular Multimodality:** LoRA-based design allows adding modalities without retraining the base model.  
- **High Efficiency:** Uses parameter-efficient tuning and frozen backbone strategy.  
- **Scalability:** Supports context lengths up to 128 K and long-duration audio (>2 hours).  
- **Quality over Quantity:** Relies on carefully curated, verified data mixtures for all modalities.

---

## 📚 Citation

> Microsoft Research.  
> **Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs.**  
> *arXiv preprint arXiv:2503.01743 (2025).*
