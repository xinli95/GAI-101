# ViT, CLIP, EVA, and SigLIP — Vision Encoder Foundations for Multimodal Models

This guide walks through the evolution from the **Vision Transformer (ViT)** to **CLIP**, **EVA**, and **SigLIP** — the key visual encoders used in modern multimodal architectures like BLIP‑2, Qwen‑VL, Gemini, and InternVL.

A runnable PyTorch sketch (`vit_clip_siglip_sketch.py`) accompanies this tutorial.

---

## 1) Vision Transformer (ViT)

**Paper:** _An Image is Worth 16×16 Words_ (Dosovitskiy et al., ICLR 2021)

### Core Idea

Treat an image as a sequence of fixed‑size patches, embed them, and run a **Transformer encoder** exactly like in NLP.

### Pipeline

1. **Patchify:** Split image 224×224×3 into 16×16 patches → 196 patches.
2. **Linear Projection:** Flatten each patch (16×16×3) → Linear layer → D‑dim embedding.
3. **Positional Embedding:** Add 2D position encodings.
4. **Transformer Blocks:** Self‑attention + MLP + residuals.
5. **Classification Head:** Use the `[CLS]` token representation.

### Pros / Cons

| Pros                        | Cons                              |
| --------------------------- | --------------------------------- |
| Global receptive field      | Weak inductive bias               |
| Simple architecture         | Needs large data (e.g., JFT‑300M) |
| Great for transfer learning | Slow on small datasets            |

---

## 2) CLIP — Vision–Language Alignment

**Paper:** _Learning Transferable Visual Models From Natural Language Supervision_ (OpenAI 2021)

### Objective

Train **two encoders**, one for image $f_\theta(I)$ and one for text $g_\phi(T)$, such that **matching pairs have similar embeddings**.

### Loss (InfoNCE / Symmetric Contrastive)

$$

L = -\frac{1}{N} \sum_i \left[\log\frac{e^{sim(f(I_i), g(T_i))/\tau}}{\sum_j e^{sim(f(I_i), g(T_j))/\tau}} + \log\frac{e^{sim(g(T_i), f(I_i))/\tau}}{\sum_j e^{sim(g(T_i), f(I_j))/\tau}}\right]


$$

### Highlights

- **Encoders:** ViT‑B/32, ViT‑L/14 (vision) + GPT‑like Transformer (text).
- **Training Data:** 400 M noisy (image, caption) pairs.
- **Output:** Shared 512‑D latent space.

### Strengths

- Excellent **zero‑shot transfer**: compute image embedding, compare to prompt embeddings (“a photo of a dog”).
- Reusable frozen vision backbone for downstream multimodal models.

---

## 3) EVA / EVA‑CLIP — Enhanced Vision Backbone

**Paper:** _EVA: Exploring the Limits of Masked Visual Representation Learning at Scale_ (BAAI 2023)

### Two‑Stage Training

1. **Stage 1: Masked Autoencoder (MAE) pretraining**

   - Mask 75% of image patches.
   - Train ViT to reconstruct them → strong self‑supervised visual features.

2. **Stage 2: CLIP‑style fine‑tuning**
   - Pair with text encoder, train contrastively on large corpus.

### Why Better Than CLIP

- Warm‑started from MAE → richer local features.
- Larger, cleaner data and stronger regularization.
- Supports giant backbones (ViT‑G/14, ViT‑e).

EVA‑CLIP achieves state‑of‑the‑art alignment and is used in **Qwen‑VL, InternVL, EVA‑02** etc.

---

## 4) SigLIP — Simpler & More Stable Contrastive Loss

**Paper:** _Sigmoid Loss for Language–Image Pre‑training_ (Google 2024)

### Problem in CLIP

Softmax‑based contrastive loss couples all pairs in a batch → unstable and expensive for multi‑caption or multi‑image datasets.

### SigLIP Fix: Independent Sigmoid Regression

Treat each pair independently as a binary classification task.

$$
L = -\frac{1}{N^2}\sum_{i,j}\big[y_{ij}\log\sigma(s_{ij}) + (1-y_{ij})\log(1-\sigma(s_{ij}))\big]
$$

where $$s_{ij} = \text{cosine}(f(I_i), g(T_j))/\tau$$.

### Benefits

✅ Batch‑size‑independent, more stable gradients.  
✅ Natural support for multi‑caption or multi‑view data.  
✅ Better recall and alignment quality.

Used in **Gemini 1.5**, **PaLI‑3**, **Qwen 3‑VL**.

---

## 5) Evolution Summary

| Model  | Pretraining Type | Loss                         | Architecture        | Key Use                      |
| ------ | ---------------- | ---------------------------- | ------------------- | ---------------------------- |
| ViT    | Supervised       | Cross‑entropy                | Transformer encoder | Vision backbone              |
| CLIP   | Contrastive      | Softmax InfoNCE              | Dual encoders       | Vision–language alignment    |
| EVA    | MAE + CLIP       | Reconstruction + Contrastive | Large ViT           | High‑quality visual features |
| SigLIP | CLIP variant     | Independent sigmoid loss     | Dual encoders       | Stable large‑scale training  |

---

## 6) Companion Code

`vit_clip_siglip_sketch.py` provides an inspectable PyTorch demo:

- Patchify + ViT forward
- CLIP‑style dual encoders
- Contrastive vs Sigmoid losses

Run:

```
python vit_clip_siglip_sketch.py
```

---
