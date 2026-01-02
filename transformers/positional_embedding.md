# Positional Encoding, RoPE, YaRN, and LongRoPE – A Practical Guide

This note summarizes our discussion about positional encodings in Transformers, Rotary Positional Embedding (RoPE), and its long‑context variants (YaRN, LongRoPE), including the clarifying questions that usually confuse readers.

---

## 1. Why Positional Encoding Matters

Self‑attention by itself is **permutation‑invariant**: if you shuffle the tokens, the attention mechanism has no built‑in notion of order.

To give a Transformer a sense of sequence, we inject **positional information** into token representations. That’s what positional encodings (PEs) do.

---

## 2. Classic Sinusoidal Positional Encoding

In the original Transformer, each position $p$ is assigned a deterministic vector:

- For even dimensions (index $2i$):

  $$\text{PE}(p, 2i) = \sin\bigg( \frac{p}{10000^{\frac{2i}{d}}} \bigg)$$

- For odd dimensions (index $2i+1$):

  $$\text{PE}(p, 2i+1) = \cos\bigg( \frac{p}{10000^{\frac{2i}{d}}} \bigg)$$

Here:

- $p$: position (0, 1, 2, …)
- $i$: dimension index
- $d$: model dimension

The final token representation is:

$$x'_p = x_p + \text{PE}(p)$$

Key ideas:

- Each dimension uses a **different frequency** (the denominator $10000^{2i/d}$).
- Combining many sine/cosine waves gives a **positional fingerprint** for each $p$.
- The model can infer **relative positions** (e.g., distance between two tokens) as a function of these waves.

Limitations:

- Encodes position **additively** and absolutely.
- Generalization to much longer sequences than seen in training is limited.

---

## 3. RoPE: Rotary Positional Embedding

RoPE (Rotary Positional Embedding, from _RoFormer_) takes a different approach:

> Instead of _adding_ positional vectors, it **rotates** query and key vectors in a position‑dependent way.

### 3.1 Rotation in 2D Pairs

Split the embedding into 2D pairs: $(x_{2i}, x_{2i+1})$.

For a token at position $p$, RoPE rotates each pair by an angle $\theta\_{p,i}$:

$$
\begin{bmatrix}
x'_{2i} \\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta_{p,i} & -\sin \theta_{p,i} \\
\sin \theta_{p,i} & \cos \theta_{p,i}
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$

with

$$\theta_{p,i} = \frac{p}{10000^{\frac{2i}{d}}}$$

The same rotation is applied to both queries $Q$ and keys $K$ before computing attention.

### 3.2 Why 2D pairs and “frequencies”?

The exponent $2i/d$ controls how fast $\theta_{p,i}$ grows with $p$:

- Small $i$ → small denominator → angle grows **fast** with $p$ → **high frequency**.
- Large $i$ → large denominator → angle grows **slowly** → **low frequency**.

Each 2D pair behaves like a **clock hand** with its own rotation speed.  
The **combination** of many hands (fast + slow) gives each position a unique, multi‑scale positional signature.

> 🔍 **Reader question:**  
> _If we already have sin/cos in classic PE, why switch to rotations?_

Because with rotations, the attention dot product between positions $p$ and $q$ becomes:

$$
(Q_p')^T K_q' = Q^T R_{(q-p)} K
$$

It depends on **relative position** $(q - p)$ directly. RoPE thus encodes relative positions in a cleaner, more geometric way than additive PE.

---

## 4. Why Rotate Pairs Instead of the Whole Embedding?

> 🔍 **Reader question:**  
> _Wouldn’t it be simpler to rotate the whole embedding by one angle per position?_

If we used a **single angle** $\theta_p$ for the entire vector:

- All dimensions would rotate together (only one “frequency”).
- After the angle increases by $2\pi$, the pattern repeats.
- Many positions would end up sharing the same rotated representation, which is too weak for long sequences.

By rotating **each 2D pair separately** with different speeds:

- Each pair has its own periodicity.
- The **joint pattern** across all pairs rarely repeats.
- This is like having many independent clock hands; the exact configuration is almost always unique over a large range of positions.

So 2D‑pair rotation is crucial for having rich, multi‑frequency positional information.

---

## 5. The Wrap‑Around Problem

Sine and cosine are periodic:

$$
\sin(\theta) = \sin(\theta + 2\pi), \quad
\cos(\theta) = \cos(\theta + 2\pi)
$$

For each pair $i$, the rotation angle is:

$$
\theta_{p,i} = \frac{p}{10000^{\frac{2i}{d}}}
$$

This pair’s rotation pattern **repeats** when $\theta\_{p,i}$ increases by $2\pi$.  
The position where this happens is:

$$
p_{\text{wrap}, i} = 2\pi \cdot 10000^{\frac{2i}{d}}
$$

- For **low $i$** (high frequency), $p\_{\text{wrap}, i}$ is small (fast repetition).
- For **high $i$** (low frequency), $p\_{\text{wrap}, i}$ is large (slow repetition).

> 🔍 **Reader question:**  
> _If one pair wraps every ~6 tokens, doesn’t that mean we’re already “done” after 6 tokens?_

No, because that’s just one pair. Other pairs have different $p\_{\text{wrap},i}$.  
The **full positional encoding** is the concatenation of _all_ these rotating 2D pairs.  
Even if one pair repeats, the others have not, so the combined pattern across all dimensions still distinguishes positions.

This is analogous to a real clock:

- The second hand repeats every 60 seconds.
- But the **full configuration** (hour, minute, second hands) repeats only every 12 hours.

### When does this become a real problem?

As $p$ becomes very large (e.g., $\gg 4\text{K}$), many **mid‑frequency** pairs will have completed several full rotations.  
Then multiple different positions can share very similar overall phase patterns.

Empirically, for typical LLM settings (e.g. LLaMA‑style models trained with context 4K):

- Positional encodings remain well‑behaved for up to about **8K–16K** tokens.
- Beyond **16K–32K** tokens, aliasing (wrap‑around effects) start to noticeably degrade attention and model quality.

That’s what people mean by “RoPE breaks down around 16K–32K” without modification.

---

## 6. Extending Context: RoPE Variants

To safely extend context lengths beyond what the model was trained on, people modify **how the angle $\theta_{p,i}$ is computed**.

The general idea:

$$
\theta_{p,i} = \frac{f(p)}{10000^{\frac{2i}{d}}}
$$

where $f(p)$ is a **remapped** position that grows more slowly than $p$ at large values.

### 6.1 NTK‑RoPE (used in LLaMA‑2/3)

NTK‑RoPE effectively changes the base from 10000 to a larger value $b\_{\text{ntk}}$:

$$
\theta'_{p,i} = \frac{p}{b_{\text{ntk}}^{\frac{2i}{d}}}, \quad b_{\text{ntk}} > 10000
$$

This slows down angle growth, delaying wrap‑around and enabling longer context (e.g., 16K).

---

### 6.2 YaRN (Yet another RoPE extension)

YaRN is designed to extend context efficiently **without retraining from scratch**.

Assume the model was trained with context length $L\_{\text{orig}}$ (e.g., 4K).  
YaRN defines a compressed position $p'$:

- For positions inside the training window:

  $$p' = p, \quad p \le L_{\text{orig}}$$

- For positions beyond that:

  $$p' = L_{\text{orig}} + \alpha (p - L_{\text{orig}}), \quad p > L_{\text{orig}}$$

with $0 < \alpha < 1$ a compression factor.

Then:

$$
\theta_{p,i} = \frac{p'}{10000^{\frac{2i}{d}}}
$$

Properties:

- **Backward compatible**: for $p \le L\_{\text{orig}}$, RoPE behaves exactly as in training.
- **Slower growth** beyond $L\_{\text{orig}}$: rotations don’t wrap as quickly, allowing stable attention at 32K, 64K, or 128K tokens.

---

### 6.3 LongRoPE

LongRoPE generalizes the idea of YaRN:

- Replaces the piecewise linear mapping with a **smooth nonlinear function** $f(p)$ that grows sublinearly for large $p$.
- Optionally makes $f(p)$ **learnable** during fine‑tuning.
- Can include **layer‑wise** scaling (deeper layers get gentler positional effects).

Goal: preserve local behavior within the original training window while supporting ultra‑long contexts (128K–1M tokens).

---

## 7. When Are These Applied? Training vs Inference

> 🔍 **Reader question:**  
> _Is RoPE or YaRN applied during training or only at inference? How does “mathematical extrapolation” work?_

Key points:

- **RoPE itself** is a deterministic function; it has **no learned parameters**.
- During pre‑training, the model uses some base RoPE (e.g., with sequence up to 4K).
- Later, at **inference or fine‑tuning time**, you can _change the function_ that maps $p$ to $\theta_{p,i}$.

Concretely, in code, you are changing something like:

```python
# training-time RoPE
theta = pos / (10000 ** (2 * i / dim))
```

to:

```python
# extended RoPE (e.g., NTK-RoPE or YaRN)
pos_scaled = f(pos)              # e.g., compressed or rescaled position
theta = pos_scaled / (10000 ** (2 * i / dim))
```

The rest of the model (weights, architecture) stays the same.

This is what we mean by:

- **“Extrapolation is done mathematically”**:  
  We extend the usable range of positional encodings by redefining the **angle mapping function** $f(p)$, without retraining the entire model.

Then:

- Optionally, a **short fine‑tuning** phase on long sequences can further adapt the model to the new positional geometry.

---

## 8. Modern Practice (High Level)

Modern LLM recipe (LLaMA, Qwen, etc.) typically looks like:

1. **Pre‑train** with standard RoPE and a base context window (e.g., 4K or 8K).
2. **Extend context** by replacing the angle computation with NTK‑RoPE, YaRN, or LongRoPE (i.e., redefine $f(p)$).
3. **Fine‑tune** lightly on long‑context data to stabilize behavior (especially for 32K–128K or more).

Examples (roughly):

- LLaMA‑2: 4K → 16K using NTK‑scaled RoPE.
- Qwen‑2 / Qwen‑3: 4–8K → 32–128K using YaRN + light long‑context fine‑tuning.
- LongRoPE: 4K → 128K–1M using a more flexible $f(p)$ and continued training.

---

## 9. Summary

- Transformers need positional encoding because attention alone has no notion of order.
- Sinusoidal PE adds sin/cos values; RoPE rotates embeddings in 2D subspaces.
- Rotating **each 2D pair** with different frequencies lets the model represent positions uniquely over long ranges.
- Because sin/cos are periodic, RoPE eventually suffers from **wrap‑around** (aliasing) at very long contexts ($~16\text{K}–32\text{K}$ for typical models).
- Variants like **NTK‑RoPE**, **YaRN**, and **LongRoPE** remap positions to **slow down** angle growth and safely extend context length.
- These extensions are often applied **after pre‑training**, by changing the angle‑mapping function analytically (sometimes followed by light fine‑tuning).

This is the core intuition and modern practice around positional encoding and RoPE‑style long‑context extensions.
