# BLIP & BLIP‑2 — A Practical, Technical Guide

This mini‑tutorial explains how **BLIP (Bootstrapped Language–Image Pre‑training)** and **BLIP‑2** work, why they’re designed this way, and how the **Q‑Former** bridges a frozen vision encoder (e.g., CLIP ViT) to a frozen LLM. It also ships with a compact PyTorch sketch you can run to see the data flow.

---

## 1) TL;DR

- **BLIP (2022)** jointly trains vision and text components with three objectives:

  - **ITC** (Image–Text Contrastive): global alignment like CLIP.
  - **ITM** (Image–Text Matching): pair‑level discrimination with cross‑attention.
  - **LM** (Language Modeling): conditional captioning / QA generation.
  - **Bootstrapping**: generate _better captions_ (pseudo‑labels) for noisy web data and retrain on those higher‑quality pairs.

- **BLIP‑2 (2023)** reuses strong **frozen** backbones (a vision encoder + a large LLM) and trains only a lightweight **Q‑Former** (+ projector) to compress vision tokens into a small set of **learned query tokens** consumable by the LLM. This yields efficient training and strong performance with modest compute.

---

## 2) BLIP: What’s Inside

**High‑level pipeline**

```
Image ──► ViT/Swin (vision encoder) ──► vision tokens
Text  ──► BERT‑like (text encoder/decoder)

[ITC] dual encoders for global alignment
[ITM] cross‑encoder predicts if (image, text) match
[LM ] decoder learns to generate captions/answers, conditioned on image tokens
```

### 2.1 Objectives

1. **ITC (contrastive)** — like CLIP
   - Encode image `f(I)` and text `g(T)` to the same space.
   - InfoNCE loss pulls matched (I,T) together; pushes mismatched apart.
2. **ITM (matching)** — cross‑encoder
   - Concatenate image & text tokens, run cross‑attention, classify **match / non‑match**.
   - Teaches _fine‑grained_ grounding beyond global ITC.
3. **LM (language modeling)** — generative
   - Condition on image tokens; predict next word for caption/QA.
   - Gives the model _generation_ ability, not just recognition.

These are often **interleaved** during training so the model learns aligned, grounded, and generative behaviors together.

### 2.2 Bootstrapping in BLIP

Web image–text pairs are noisy. **Bootstrapping** means:

1. Train an initial model; 2) Use it to **generate cleaner/better captions**; 3) Filter the generated captions; 4) **Retrain** on these refined pairs.  
   This iterative loop improves data quality without manual labeling.

---

## 3) BLIP‑2: Why Q‑Former?

Large LLMs + high‑res ViTs are expensive to train end‑to‑end. BLIP‑2 freezes both and inserts a **Q‑Former** in the middle.

**Core idea**: keep the **vision encoder frozen** (e.g., CLIP ViT‑L/14) that outputs `N` patch tokens; learn a small module that compresses them into `M` **query tokens** (`M << N`) that the **frozen LLM** can ingest.

```
Image ──► (frozen) ViT ──► V = {v1..vN}   --vision tokens
                 ▲
      learnable queries Q = {q1..qM}       --parameters
                 │
        Q‑Former (self‑attn on Q + cross‑attn Q↔V)
                 ▼
       summarized tokens Q' (M×Dq) ──► linear projector ──► LLM embed dim
                 ▼
                LLM
```

### 3.1 Learnable Query Tokens

- A **fixed bank of parameters** (e.g., M=32 vectors), _not derived_ from the input.
- Through **cross‑attention**, each query learns to “look at” different parts/aspects of the image tokens and becomes a **visual summary token**.
- Why not pool? A single pooled vector loses object/region diversity; multiple queries let the model preserve **disentangled** axes of information.

### 3.2 Q‑Former Block

Each layer typically does:

1. **Self‑attention on queries** (helps them coordinate/reduce redundancy).
2. **Cross‑attention (queries attend to vision tokens)**.
3. **Feed‑forward (MLP)** with residuals + LayerNorm.

Only the Q‑Former and a small **projection layer** to the LLM embedding space are trained; **ViT and LLM stay frozen**.

---

## 4) Training Cheatsheet

### BLIP (joint training)

- **Backbones**: ViT/Swin + BERT‑like text model.
- **Losses**: ITC + ITM + LM (mixture per step or per‑batch).
- **Data**: Large web corpora (e.g., CC12M/LAION); optionally use **bootstrapped captions**.
- **Use cases**: captioning, VQA, retrieval, grounding.

### BLIP‑2 (frozen backbones + small bridge)

- **Backbones**: Frozen CLIP ViT‑L/14 (or EVA/SigLIP) + frozen LLM (e.g., Flan‑T5/Vicuna).
- **Train**: Q‑Former + projector (and later light instruction tuning).
- **Losses**: ITM + LM (caption/QA), optionally contrastive variants.
- **Benefits**: strong results with a fraction of compute; easy to swap vision/LLM backbones.

---

## 5) Practical Tips

- Start with a proven **vision encoder** (CLIP/EVA/SigLIP) to get robust patch embeddings.
- Keep **M** (queries) modest (e.g., 16–64). More queries ≠ always better.
- When memory‑bound: reduce image resolution, use gradient checkpointing, or distill token counts.
- For instruction‑following VLMs: first train the **bridge** (Q‑Former + projector), then do a small **SFT** on multimodal instructions.

---

## 6) Companion Code

See `blip_qformer_sketch.py` for a compact PyTorch implementation of:

- A minimal **Q‑Former** (self‑attn on queries + cross‑attn to vision tokens).
- A small **projection** to an LLM embedding space.
- A toy training step (replace with ITM/LM losses in practice).

```
python blip_qformer_skretch.py
```

You’ll see shapes like:

```
llm_visual_tokens: torch.Size([2, 32, 4096])
loss: 1.23
```

---

## 7) Glossary

- **ITC**: Image–Text Contrastive objective (global alignment, CLIP‑style).
- **ITM**: Image–Text Matching (binary match classification with cross‑attention).
- **LM**: Language Modeling (causal or seq2seq generation).
- **Q‑Former**: Query‑centric transformer that summarizes frozen vision tokens into a few learned query tokens suitable for LLMs.

---

## 8) References (for deeper reading)

- **BLIP**: Bootstrapped Language‑Image Pre‑training for Unified Vision‑Language Understanding and Generation (2022).
- **BLIP‑2**: Bootstrapping Language‑Image Pre‑training with Frozen Image Encoders and Large Language Models (2023).
- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision (2021).
- **ViT**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021).
- **SigLIP**: Sigmoid Loss for Language‑Image Pre‑training (2024).
- **MAE/EVA**: Masked Autoencoders Are Scalable Vision Learners (2021); EVA (2023).
