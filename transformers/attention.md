# 🔍 Understanding Attention Mechanisms in Modern Transformers

**From Multi-Head to Multi-Latent Attention**

---

## 1. What is “attention”?

At its core, attention is a **content-based retrieval** mechanism.  
For every token, the model asks:

> “Given what I know (my hidden state), which other tokens in the sequence are relevant to me?”

Mathematically, for queries $Q$, keys $K$, and values $V$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- **Q (Query)** – representation of the _current token_
- **K (Key)** – representations of _context tokens_
- **V (Value)** – the _information_ attached to each key

---

## 2. Multi-Head Attention (MHA)

Introduced in _Attention Is All You Need (Vaswani et al., 2017)_,  
MHA lets the model look at information from multiple “perspectives”.

Each head has its own projection matrices $W_Q, W_K, W_V$.  
All heads attend separately, then their outputs are concatenated.

$$
\text{MHA}(X) = \text{Concat}(h_1,\ldots,h_H) W_O
$$

**Pros**

- High expressivity — each head can learn different relations.

**Cons**

- Expensive memory and compute: every head stores its own K/V cache.
- Inference KV cache grows as $O(L \times H \times d_{\text{head}})$.

**Used in:** BERT, GPT-2, early LLaMA.

---

## 3. Multi-Query Attention (MQA)

**Goal:** Reduce KV memory at inference.

Instead of separate $K,V$ for every head, MQA **shares one global pair** across all heads.

$$
K = X W_K, \quad V = X W_V
$$

All heads use the same $(K,V)$ but different $W_Q$.

**Result:**  
KV cache is **1×** (instead of H×) → huge memory savings.

**Trade-off:**  
Slight drop in quality since all heads look at identical K/V.

**Used in:** PaLM, Gemini, Claude models.

---

## 4. Grouped Query Attention (GQA)

**Goal:** Balance between MHA expressivity and MQA efficiency.

Queries are split into groups; each group shares one $(K,V)$ pair.

Example:

- 64 query heads
- 8 groups → each group handles 8 query heads → 8 × smaller KV cache.

**Benefits**

- Almost no quality loss vs. MHA
- 8–16× faster inference vs. full MHA

**Used in:** LLaMA-2/3, Qwen-2, Phi-3, Mistral.

---

## 5. Why all these optimizations target _K_ and _V_

During autoregressive generation:

1. Each new token computes its own **Q, K, V**.
2. **Only K and V are cached** for reuse by future tokens.
3. Q is temporary and recomputed each step.

→ Optimizing Q doesn’t help inference speed or memory,  
but reducing or sharing K/V drastically does.

---

## 6. Multi-Head Latent Attention (MLA)

**Introduced by:** DeepSeek-V2/V3 (2024–2025)  
**Goal:** Push KV efficiency even further.

Instead of keeping full-dimensional K/V, MLA projects them into a **compact latent space**:

$$
K_{\text{latent}} = K W_{K\rightarrow L}, \quad
V_{\text{latent}} = V W_{V\rightarrow L}
$$

Attention operates in this low-dimensional latent space, and results are projected back to the model dimension.

**Effect:**

- Reduces KV cache by another 8–16× beyond GQA
- Maintains or improves perplexity due to learned latent mixing

**Used in:** DeepSeek-V2, V3 (state-of-the-art long-context LLMs)

---

## 7. Comparative Overview

| Mechanism | KV Sharing               | Latent Compression | KV Cache Size | Expressivity | Typical Use     |
| --------- | ------------------------ | ------------------ | ------------- | ------------ | --------------- |
| **MHA**   | None                     | No                 | Very large    | ⭐⭐⭐⭐     | GPT-2, BERT     |
| **MQA**   | All heads share 1 KV     | No                 | Tiny          | ⭐⭐         | PaLM            |
| **GQA**   | Groups of heads share KV | No                 | Small         | ⭐⭐⭐⭐     | LLaMA-3, Qwen-2 |
| **MLA**   | Full latent projection   | Yes                | Tiny-tiny     | ⭐⭐⭐⭐     | DeepSeek-V2/V3  |

---
