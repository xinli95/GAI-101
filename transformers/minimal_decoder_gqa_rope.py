"""
toy_decoder_gqa_rope.py

A tiny, but realistic, decoder-only Transformer using:

- Token embeddings
- Grouped Query Attention (GQA)
- Rotary Positional Embedding (RoPE) on Q/K
- RMSNorm + SwiGLU MLP
- Multi-layer decoder with per-layer KV cache
- Equivalence test:
    full forward (with causal mask) vs
    incremental decode (one token at a time using KV cache)

This is structurally similar to modern LLM decoders, but shrunk down.
"""

import math
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm (LLaMA-style)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(norm + self.eps)
        return self.weight * x


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Standard RoPE implementation (like in LLaMA/Qwen).
    dim: per-head dimension (Dh)
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin(self, seq_len: int, offset: int, device=None):
        if device is None:
            device = self.inv_freq.device

        # positions: [T]
        positions = torch.arange(offset, offset + seq_len, device=device).float()
        # freqs: [T, dim/2]
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        # [T, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]  # [1,1,T,dim]
        sin = emb.sin()[None, None, :, :]  # [1,1,T,dim]
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [B, H, T, Dh], cos/sin: [1,1,T,Dh]
    """

    def rotate_half(y: torch.Tensor) -> torch.Tensor:
        y1 = y[..., : y.size(-1) // 2]
        y2 = y[..., y.size(-1) // 2 :]
        return torch.cat([-y2, y1], dim=-1)

    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA) with RoPE
# ---------------------------------------------------------------------------

class GQAWithRoPE(nn.Module):
    """
    Grouped Query Attention + RoPE on Q and K.

    - num_heads: total query heads
    - num_kv_heads: how many KV "groups"
        * each KV head is shared by group_size = num_heads / num_kv_heads Q heads
    - rope: RotaryEmbedding applied to Q and K per head

    KV cache format (per layer):
        {
          "k": [B, H_kv, T_past, Dh],
          "v": [B, H_kv, T_past, Dh]
        }
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        rope: RotaryEmbedding,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def _reshape_q(self, q: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.size()
        q = q.view(B, T, self.num_heads, self.head_dim)
        return q.transpose(1, 2)  # [B, H, T, Dh]

    def _reshape_kv(self, proj: torch.Tensor) -> torch.Tensor:
        B, T, _ = proj.size()
        kv = proj.view(B, T, self.num_kv_heads, self.head_dim)
        return kv.transpose(1, 2)  # [B, H_kv, T, Dh]

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        x: [B, T, d_model]
        kv_cache: per-layer cache dict or None
        attn_mask: [1,1,T,T_total] or None
        """
        B, T, _ = x.shape

        # Project
        q = self._reshape_q(self.W_q(x))       # [B, H,   T, Dh]
        k_new = self._reshape_kv(self.W_k(x))  # [B, H_kv, T, Dh]
        v_new = self._reshape_kv(self.W_v(x))  # [B, H_kv, T, Dh]

        # Determine past length for RoPE
        if kv_cache is not None:
            past_len = kv_cache["k"].size(2)
        else:
            past_len = 0

        cos, sin = self.rope.get_cos_sin(seq_len=T, offset=past_len, device=x.device)

        # Apply RoPE to current chunk
        q = apply_rotary_pos_emb(q, cos, sin)       # [B, H,   T, Dh]
        k_new = apply_rotary_pos_emb(k_new, cos, sin)  # [B, H_kv, T, Dh]

        # Append to cache
        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k_new], dim=2)  # [B, H_kv, T_total, Dh]
            v = torch.cat([kv_cache["v"], v_new], dim=2)
        else:
            k, v = k_new, v_new

        new_cache = {"k": k, "v": v} if use_cache else None

        # Repeat KV heads to match Q heads
        k_tile = k.repeat_interleave(self.group_size, dim=1)  # [B, H, T_total, Dh]
        v_tile = v.repeat_interleave(self.group_size, dim=1)  # [B, H, T_total, Dh]

        Dh = self.head_dim
        att = torch.matmul(q, k_tile.transpose(-2, -1)) / math.sqrt(Dh)  # [B, H, T, T_total]

        if attn_mask is not None:
            att = att + attn_mask

        p = F.softmax(att, dim=-1)
        ctx = torch.matmul(p, v_tile)  # [B, H, T, Dh]

        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.d_model)  # [B,T,d_model]
        out = self.W_o(ctx)
        return out, new_cache


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    Simplified SwiGLU MLP:
      FFN(x) = W_o( (W_gate(x) * swish(W_up(x))) )

    Typically hidden_dim ~ 4 * d_model, but we can use a smaller multiple.
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_up = nn.Linear(d_model, hidden_dim)
        self.w_gate = nn.Linear(d_model, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.w_up(x)
        gate = self.w_gate(x)
        return self.w_out(F.silu(up) * gate)


# ---------------------------------------------------------------------------
# One decoder block: RMSNorm -> GQA+RoPE -> RMSNorm -> MLP, with residuals
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden_dim: int,
        rope: RotaryEmbedding,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GQAWithRoPE(d_model, num_heads, num_kv_heads, rope)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Attention
        h = self.attn_norm(x)
        h, new_cache = self.attn(h, kv_cache=kv_cache, use_cache=use_cache, attn_mask=attn_mask)
        x = x + h

        # MLP
        h2 = self.mlp_norm(x)
        h2 = self.mlp(h2)
        x = x + h2

        return x, new_cache


# ---------------------------------------------------------------------------
# Tiny decoder model (stack of DecoderBlocks + embeddings + LM head)
# ---------------------------------------------------------------------------

class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        mlp_hidden_dim: int = 1024,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # Token embeddings only; no absolute pos emb (RoPE handles positions)
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Shared RoPE module (per-head dim)
        rope = RotaryEmbedding(dim=d_model // num_heads)

        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                rope=rope,
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        input_ids: [B, T]
        kv_caches: list of length num_layers, each either None or dict for that layer

        Returns:
            logits: [B, T, vocab_size]
            new_kv_caches: list of caches (if use_cache=True) else None
        """
        B, T = input_ids.shape
        device = input_ids.device

        if kv_caches is None:
            kv_caches = [None] * self.num_layers

        x = self.tok_emb(input_ids)  # [B, T, d_model]

        new_caches = [] if use_cache else None

        for layer, cache in zip(self.layers, kv_caches):
            x, new_cache = layer(x, kv_cache=cache, use_cache=use_cache, attn_mask=attn_mask)
            if use_cache:
                new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches


# ---------------------------------------------------------------------------
# Helper: causal mask
# ---------------------------------------------------------------------------

def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]


# ---------------------------------------------------------------------------
# Equivalence test: full forward vs incremental decoding with KV cache
# ---------------------------------------------------------------------------

def test_full_vs_incremental():
    torch.manual_seed(0)

    vocab_size = 100
    d_model = 256
    num_layers = 2
    num_heads = 8
    num_kv_heads = 2
    mlp_hidden_dim = 512
    T = 8
    B = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TinyDecoderLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_hidden_dim=mlp_hidden_dim,
    ).to(device)

    input_ids = torch.randint(0, vocab_size, (B, T), device=device)

    # 1) Full forward with causal mask
    full_mask = causal_mask(T, device=device)
    logits_full, _ = model(input_ids, kv_caches=None, use_cache=False, attn_mask=full_mask)
    # [B, T, vocab_size]

    # 2) Incremental decoding, 1 token at a time, updating caches
    kv_caches = None
    logits_inc_list = []

    for t in range(T):
        step_ids = input_ids[:, t : t+1]  # [B, 1]

        logits_step, kv_caches = model(
            step_ids,
            kv_caches=kv_caches,
            use_cache=True,
            attn_mask=None,  # causal is enforced by cache
        )
        # logits_step: [B, 1, vocab_size]
        logits_inc_list.append(logits_step)

        # Inspect cache growth for the first layer only
        if t == 0 or t == T - 1:
            print(f"\nStep {t}:")
            layer0_cache = kv_caches[0]
            print("  layer0 k.shape:", tuple(layer0_cache["k"].shape))
            print("  layer0 v.shape:", tuple(layer0_cache["v"].shape))

    logits_inc = torch.cat(logits_inc_list, dim=1)  # [B, T, vocab_size]

    # Compare
    max_abs_diff = (logits_full - logits_inc).abs().max().item()
    print(f"\nMax |logits_full - logits_inc| = {max_abs_diff:.6e}")


if __name__ == "__main__":
    test_full_vs_incremental()
