# InternVL3 Vision–Language Pipeline

### _From Image → Vision Embeddings → LLM Generation (Step-by-Step)_

This document explains the **entire data flow** inside InternVL3 (1B/8B/9B/14B), following the actual `modeling_internvl_chat.py` implementation from the official HuggingFace release.

It is written for readers without strong computer vision background.

---

# Table of Contents

1. [Introduction](#introduction)
2. [High-Level Pipeline Overview](#high-level-pipeline-overview)
3. [Step 1 — Preprocessing the Image](#step-1--preprocessing-the-image)
4. [Step 2 — Tiling the Image (Dynamic Preprocess)](#step-2--tiling-the-image-dynamic-preprocess)
5. [Step 3 — Converting Tiles into pixel_values](#step-3--converting-tiles-into-pixel_values)
6. [Step 4 — Extracting Vision Features (extract_feature)](#step-4--extracting-vision-features-extract_feature)
   - 4.1 Running the Vision Encoder
   - 4.2 Removing the CLS Token
   - 4.3 Reshaping into a 2D grid
   - 4.4 Pixel Shuffle Downsampling
   - 4.5 Flatten → Sequence of Vision Tokens
   - 4.6 MLP Projection into LLM Embedding Space
7. [Step 5 — Injecting Vision Tokens into the LLM Input](#step-5--injecting-vision-tokens-into-the-llm-input)
8. [Step 6 — Positional Encoding with RoPE](#step-6--positional-encoding-with-rope)
9. [Step 7 — LLM Forward & Generation](#step-7--llm-forward--generation)
10. [End-to-End Diagram](#end-to-end-diagram)
11. [Appendix — Annotated Pseudocode](#appendix--annotated-pseudocode)

---

# Introduction

InternVL3 is a **unified multimodal model** combining:

- a **vision encoder** (ViT/EVA backbone),
- a **token alignment module** (pixel shuffle + MLP projection),
- a **Qwen2/Qwen3 language model**,
- and a custom **image insertion mechanism** using `<IMG_CONTEXT>` tokens.

This design makes image patches behave **exactly like text tokens** once embedded, enabling the LLM to attend to vision features with standard self-attention.

---

# High-Level Pipeline Overview

```
PIL image
    ↓
dynamic_preprocess        ← tiling into multiple 448x448 crops
    ↓
pixel_values: [T, 3, 448, 448]
    ↓
Vision Encoder (ViT/EVA)
    ↓
patch embeddings: [T, 196, d_vit]
    ↓
reshape to 2D: [T, h, w, d_vit]
    ↓
pixel_shuffle downsampling
    ↓
vision tokens: [T, N, d_vit']
    ↓
MLP projection → LLM dimension
    ↓
vision tokens: [T_total, N', d_llm]
    ↓
Insert tokens into LLM sequence at <IMG_CONTEXT> positions
    ↓
RoPE positional encoding
    ↓
Qwen Transformer blocks
    ↓
Generated text
```

---

# Step 1 — Preprocessing the Image

All images are:

- converted to **RGB**
- resized to fixed **448 × 448** tiles
- normalized using ImageNet mean/std

```python
transform = Compose([
    Lambda(lambda img: img.convert("RGB")),
    Resize((448, 448)),
    ToTensor(),
    Normalize(mean, std),
])
```

This creates tensors suitable for a Vision Transformer.

---

# Step 2 — Tiling the Image (Dynamic Preprocess)

InternVL does not rely on a single image crop.

Instead, it uses `dynamic_preprocess`:

- splits the image into **1–12 tiles**
- tiles are **448 × 448**
- based on **aspect ratio**
- includes an **optional thumbnail tile**

This allows:

- OCR on tiny text
- zoom-in details for charts
- high-resolution document reading
- better spatial coverage

Results in a list of tiles:

```
[T tiles] × [3 × 448 × 448]
```

---

# Step 3 — Converting Tiles into pixel_values

After transformation:

```
pixel_values: [T, 3, 448, 448]
```

Where:

- `T` = number of tiles
- `3` = channels (RGB)
- `448 × 448` = spatial resolution

This is the direct input to InternVL’s ViT.

---

# Step 4 — Extracting Vision Features (`extract_feature`)

This corresponds to InternVL3’s internal vision encoder + projection logic.

### 4.1 Run the Vision Encoder

```python
vision_output = self.vision_model(pixel_values)
vit_embeds = vision_output.last_hidden_state   # [T, 1 + P, d_vit]
```

Where:

- `P = 196` patch tokens from a 14×14 grid based on 32 x 32 patches
- first token = CLS (will be dropped)

### 4.2 Drop the CLS Token

```python
vit_embeds = vit_embeds[:, 1:, :]   # [T, 196, d_vit]
```

### 4.3 Reshape into a 2D grid

```
vit_embeds = [T, 14, 14, d_vit]
```

This treats patches spatially like a feature map.

---

### 4.4 Pixel Shuffle Downsampling

InternVL3 uses:

```python
pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
```

With ratio ≈ **0.5**, this:

- reduces spatial size:  
  `14×14` → `7×7`
- increases channel dimension (compensating for lost resolution)

### Why?

Reduces total vision tokens → more efficient LLM context.

This is **crucial** for handling many image tiles.

---

### 4.5 Flatten back into a sequence

After pixel shuffle:

```
vit_embeds: [T, N', d_vit']
```

Where:

- `N' = 7 × 7 = 49` tokens per tile (instead of 196)
- `d_vit' = upgraded channel dimension after shuffle`

---

### 4.6 MLP Projection into LLM Embedding Space

To make vision tokens compatible with Qwen’s embeddings:

```python
vit_embeds = self.mlp1(vit_embeds)
```

Now shape:

```
vit_embeds: [T, N', d_llm]
```

This is now **in the same hidden dimension** the LLM uses.

These are the final **vision tokens**.

---

# Step 5 — Injecting Vision Tokens into the LLM Input

When you write:

```
<image>
What does the sign say?
```

the tokenizer expands `<image>` into many `<IMG_CONTEXT>` placeholder tokens:

```
<image>,
<IMG_CONTEXT>, <IMG_CONTEXT>, <IMG_CONTEXT>, ...,  (N' × T tokens)
"What", "does", "the", ...
```

InternVLChatModel replaces **exactly those** `<IMG_CONTEXT>` tokens with the vision embeddings.

Conceptually:

```
Text embedding stream:
[ embed(<IMG_CONTEXT>), embed(<IMG_CONTEXT>), ... embed("What"), embed("does"), ... ]

Replace:
embed(<IMG_CONTEXT>) → vision_token[i]
```

After replacement:

```
[ img_tok_1, img_tok_2, ..., img_tok_K, txt_tok_1, txt_tok_2, ... ]
```

This creates **one unified sequence** of LLM tokens.

---

# Step 6 — Positional Encoding with RoPE

Qwen2/Qwen3 apply **Rotary Positional Embeddings (RoPE)** to **both**:

- image tokens,
- text tokens.

Positional IDs look like:

```
0: img_tok_1
1: img_tok_2
...
K-1: last image token
K:   "What"
K+1: "does"
...
```

**InternVL does not treat image tokens differently** at inference.  
They are simply tokens with positional IDs.

Dynamic-RoPE scaling (config-level) handles long contexts.

---

# Step 7 — LLM Forward & Generation

Finally, the entire token stream flows into Qwen:

1. Embeddings →
2. Attention blocks →
3. Residual layers →
4. Language modeling head →
5. Next-token probability distribution

During generation (`model.generate` or `model.chat`):

- Qwen auto-regressively generates new tokens,
- optionally using **contrastive decoding**, **top-k**, **beam search**, etc.

Image tokens **do not appear** in generated output —  
they only serve as context for the model to reason over.

---

# End-to-End Diagram

```
+-------------------+
|       Image       |
+---------+---------+
          |
          v
   dynamic_preprocess
 (1–12 tiles, 448x448)
          |
          v
pixel_values [T, 3, 448,448]
          |
          v
+-------------------------+
|     Vision Encoder      |
|   (EVA / ViT-H backbone)|
+-------------------------+
          |
     196 patch tokens
          |
 drop CLS |
          v
   reshape to [T, 14, 14, d]
          |
          v
  Pixel Shuffle Downsample
   (reduce spatial tokens)
          |
          v
 Flatten: [T, N', d']
          |
          v
+--------------------------+
| MLP Projection           |
|  → d_llm (Qwen dimension)|
+--------------------------+
          |
vision tokens [V, d_llm]
          |
          v
LLM input embeddings
          |
   RoPE positional encoding
          |
          v
+--------------------------+
|      Qwen2 Transformer   |
+--------------------------+
          |
          v
   Next-token generation
          |
          v
       Output text
```

---

# Appendix — Annotated Pseudocode

### Key steps extracted from InternVL3 official code

```python
# 1. Preprocess input images → pixel_values
pixel_values = preprocess(image)   # shape [T, 3, 448,448]

# 2. Vision backbone
vit_out = self.vision_model(pixel_values)  # [T, 1+P, d_vit]

# 3. Drop CLS
vit_embeds = vit_out[:, 1:, :]    # [T, P, d_vit]

# 4. Reshape into 2D grid
vit_embeds = vit_embeds.reshape(T, h, w, d_vit)

# 5. Pixel shuffle downsampling
vit_embeds = pixel_shuffle(vit_embeds)  # [T, h', w', d_vit']

# 6. Flatten back to sequence
vit_embeds = vit_embeds.reshape(T, N', d_vit')

# 7. Project to LLM hidden size
vit_embeds = mlp1(vit_embeds)  # [T, N', d_llm]

# 8. Build input embedding matrix from tokenizer
input_embeds = llm_embeddings(input_ids)

# 9. Replace IMG_CONTEXT positions
input_embeds[img_positions] = vit_embeds.reshape(-1, d_llm)

# 10. Forward through Qwen2
outputs = language_model(
    inputs_embeds=input_embeds,
    position_ids=...,
    attention_mask=...,
)

# 11. Generate
generated_tokens = model.generate(...)
```

---

# ✔ Summary

InternVL3 uses a **clean and powerful architecture**:

- **Dynamic image tiling** for high OCR & detail resolution
- **ViT encoder** to extract patch features
- **PixelShuffle** downsampling for efficient token count
- **MLP projection** into LLM space
- **Token-level injection** at `<IMG_CONTEXT>` positions
- **Uniform RoPE** to mix image+text tokens in a single Transformer
- **Qwen LLM** to perform reasoning and generation

The key insight:

> **InternVL3 treats image patches as fully-fledged tokens inside the language model. This allows a pure LLM to perform multimodal reasoning with no architectural modifications other than token injection.**
