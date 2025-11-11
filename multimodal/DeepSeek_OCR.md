# DeepSeek-OCR — Architecture & Tutorial

This document walks you through the **DeepSeek-OCR** model architecture — a state-of-the-art, open-source OCR model designed for **visual token compression** — and provides a minimal PyTorch sketch illustrating how its components fit together.

---

## 🧠 1. Motivation

Traditional OCR systems convert each image patch or character into text tokens, which can explode in length for large documents.  
DeepSeek-OCR instead uses **compressed visual tokens** that still preserve layout and text fidelity.

In the paper *"DeepSeek-OCR: Contexts Optical Compression (arXiv:2510.18234)"*, the authors show that with only $\frac{1}{10}$ of the usual text-token count, OCR accuracy remains above 97%.

---

## 🧩 2. Model Overview

DeepSeek-OCR consists of three major components:

```
       +-----------------------------+
       |      DeepSeek-V2 LM         |
       |  (causal text decoder)      |
       +-------------▲---------------+
                     │
           MLP Projector (fusion)
                     │
  ┌──────────────────┴──────────────────┐
  │                                     │
  ▼                                     ▼
SAM-style ViT (global)        CLIP-like ViT (local crops)
```
Both encoders are collectively called **DeepEncoder**.

---

## ⚙️ 3. Architecture Details

### (1) Global Encoder (SAM-ViT)

- Captures **page-level layout** and long-range context.
- Operates on a full resized view (e.g. 1024×1024).
- Output shape: $(N_{global}, D_v)$

### (2) Local Encoder (CLIP-ViT)

- Processes cropped **tiles** (e.g. 640×640).
- Each tile encodes finer text regions.
- Output shape: $(N_{local}, D_v)$ per tile

### (3) Projector

- Maps vision embeddings to the LM embedding space ($D_v \rightarrow D_{llm}$).
- Supports multiple types (linear, MLP-GELU, token pooling).
- Output: visual token embeddings ready for the LLM.

### (4) Text Decoder (DeepSeek-V2)

- Standard **causal LM** generating text autoregressively.
- Receives a sequence with interleaved tokens:
  $$
  [\text{<s>}, \text{instruction}, \text{<image>}, \text{visual\_tokens}, ...]
  $$
- Predicts text or structured markup (e.g., markdown or JSON bounding boxes).

---

## 📸 4. Image Preprocessing

The image is split dynamically using a **2D cropping grid** to generate global + local views.

```python
def dynamic_preprocess(image, grid_size=(3,3), base_size=1024, crop_size=640):
    # Produce 1 global and up to 9 local crops
    global_img = image.resize((base_size, base_size))
    local_crops = []
    w, h = image.size
    cw, ch = w//grid_size[0], h//grid_size[1]
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            box = (i*cw, j*ch, (i+1)*cw, (j+1)*ch)
            local_crops.append(image.crop(box).resize((crop_size, crop_size)))
    return global_img, local_crops
```

---

## 🧮 5. Image Token Generation

Each visual feature map (after ViT) is projected and downsampled to form **image tokens**.  
If patch size = 16 and downsample ratio = 4, then number of tokens per axis is:

$$
N = \left(\frac{\text{image\_size}}{\text{patch\_size} \times \text{downsample\_ratio}}\right)^2
$$

These are inserted into the text sequence using `<image>` placeholders.

---

## 🧠 6. Training Objective

The model is trained as a **causal language model** with images interleaved in the token stream.

For a training pair $(x, I, y)$:

$$
\mathcal{L} = - \sum_t \log P_\theta(y_t | y_{<t}, x, f(I))
$$

where $f(I)$ denotes projected vision tokens.

### Datasets (per paper)
- **OmniDocBench**, **GOT-OCR2.0**, **TextVQA** (mix of doc OCR + visual-text reasoning).

---

## 🚀 7. Companion PyTorch Sketch

Below is a simplified implementation that mirrors the core idea.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Vision Encoder Blocks ---
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=512, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x

class MiniViT(nn.Module):
    def __init__(self, dim=512, depth=4, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, 2048, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return self.norm(x)

# --- Projector ---
class MLPProjector(nn.Module):
    def __init__(self, in_dim=512, out_dim=1024):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
    def forward(self, x): return self.fc(x)

# --- Simplified DeepSeek-OCR ---
class DeepSeekOCR(nn.Module):
    def __init__(self, vis_dim=512, llm_dim=1024):
        super().__init__()
        self.global_vit = MiniViT(vis_dim)
        self.local_vit = MiniViT(vis_dim)
        self.projector = MLPProjector(vis_dim, llm_dim)
        self.llm = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(llm_dim, 8, 2048, batch_first=True), num_layers=4)

    def forward(self, global_feats, local_feats, text_emb):
        vg = self.global_vit(global_feats)
        vl = self.local_vit(local_feats)
        v = self.projector(torch.cat([vg, vl], dim=1))
        out = self.llm(text_emb, v)
        return out

# --- Example ---
if __name__ == "__main__":
    B = 1
    global_feats = torch.randn(B, 196, 512)
    local_feats = torch.randn(B, 400, 512)
    text_emb = torch.randn(B, 32, 1024)
    model = DeepSeekOCR()
    out = model(global_feats, local_feats, text_emb)
    print("Output:", out.shape)
```

Output:
```
Output: torch.Size([1, 32, 1024])
```

---

## 🧭 8. Key Takeaways

| Component | Role | Example Implementation |
|------------|------|------------------------|
| Global ViT | Captures page layout | SAM-style encoder |
| Local ViT | Captures small text | CLIP-ViT encoder |
| MLP Projector | Aligns dims | GELU MLP |
| LLM (decoder) | Generates text | DeepSeek-V2 or causal LM |

---

## 📊 9. Performance Highlights (from paper)

| Metric | DeepSeek-OCR | GOT-OCR2.0 | Notes |
|---------|---------------|------------|-------|
| OCR Accuracy | **97.1%** | 96.8% | at 10× compression |
| Throughput | 200k pages/day | – | on A100-40GB |
| Compression Ratio | 10× | 1× | vs text tokens |

---

## 🔗 10. References

- DeepSeek-AI, *"DeepSeek-OCR: Contexts Optical Compression"*, arXiv:2510.18234 (2025).  
- [Hugging Face Repo](https://huggingface.co/deepseek-ai/DeepSeek-OCR)  
- [Model Code: `modeling_deepseekocr.py`](https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/modeling_deepseekocr.py)

---
