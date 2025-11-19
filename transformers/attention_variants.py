"""
attention_variants.py

A small PyTorch playground to compare different attention mechanisms:

- MHA: Multi-Head Attention (baseline)
- MQA: Multi-Query Attention (shared K/V)
- GQA: Grouped Query Attention (shared K/V per group)
- MLA (toy): Multi-Head Latent Attention (compressed latent K/V)

Run this file directly to execute:
  1) A simple single-shot test with a causal mask.
  2) An autoregressive-style cache growth demo for each variant.

    python attention_variants.py
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Multi-Head Attention (MHA)
# ---------------------------------------------------------------------------

class MHA(nn.Module):
    """
    Vanilla Multi-Head Attention (as in "Attention Is All You Need").

    - Each head has its own Q/K/V projections.
    - K/V cache size scales with num_heads.
    """

    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model] -> [B, H, T, Dh]
        """
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # [B, H, T, Dh]
        return x

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        x: [B, T, d_model]
        kv_cache: optional dict with "k" and "v" from previous steps
        attn_mask: optional additive mask broadcastable to [B, H, T, T_total]
        """
        # Project to Q/K/V and reshape into heads
        q = self._reshape_heads(self.W_q(x))  # [B, H, T, Dh]
        k = self._reshape_heads(self.W_k(x))  # [B, H, T, Dh]
        v = self._reshape_heads(self.W_v(x))  # [B, H, T, Dh]

        # Append to KV cache along time dimension (T)
        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)  # [B, H, T_total, Dh]
            v = torch.cat([kv_cache["v"], v], dim=2)  # [B, H, T_total, Dh]

        new_cache = {"k": k, "v": v} if use_cache else None

        # Scaled dot-product attention
        Dh = self.head_dim
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B, H, T, T_total]

        if attn_mask is not None:
            att = att + attn_mask

        p = F.softmax(att, dim=-1)
        ctx = torch.matmul(p, v)  # [B, H, T, Dh]

        # Merge heads
        ctx = ctx.transpose(1, 2).contiguous()  # [B, T, H, Dh]
        ctx = ctx.view(x.size(0), x.size(1), self.d_model)  # [B, T, d_model]

        out = self.W_o(ctx)  # [B, T, d_model]
        return out, new_cache


# ---------------------------------------------------------------------------
# 2. Multi-Query Attention (MQA)
# ---------------------------------------------------------------------------

class MQA(nn.Module):
    """
    Multi-Query Attention.

    - Each head has its own Q, but all heads share ONE K/V "head".
    - KV cache size does NOT depend on num_heads (only 1 KV head).
    """

    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        # K/V project to a single KV head of size head_dim
        self.W_k = nn.Linear(d_model, self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, self.head_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def _q_heads(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [B, T, d_model] -> [B, H, T, Dh]
        """
        B, T, _ = q.size()
        q = q.view(B, T, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        return q

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        q = self._q_heads(self.W_q(x))  # [B, H, T, Dh]

        # Shared KV head: [B, 1, T, Dh]
        k = self.W_k(x).unsqueeze(1)
        v = self.W_v(x).unsqueeze(1)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)  # [B, 1, T_total, Dh]
            v = torch.cat([kv_cache["v"], v], dim=2)  # [B, 1, T_total, Dh]

        new_cache = {"k": k, "v": v} if use_cache else None

        Dh = self.head_dim
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B, H, T, T_total]

        if attn_mask is not None:
            att = att + attn_mask

        p = F.softmax(att, dim=-1)
        ctx = torch.matmul(p, v)  # [B, H, T, Dh]

        ctx = ctx.transpose(1, 2).contiguous()  # [B, T, H, Dh]
        ctx = ctx.view(x.size(0), x.size(1), self.d_model)  # [B, T, d_model]

        out = self.W_o(ctx)
        return out, new_cache


# ---------------------------------------------------------------------------
# 3. Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------

class GQA(nn.Module):
    """
    Grouped Query Attention.

    - num_heads: total query heads
    - num_kv_heads: how many KV "groups"
        * each KV head is shared by group_size = num_heads / num_kv_heads Q heads
    - MQA = special case with num_kv_heads = 1
    - MHA = special case with num_kv_heads = num_heads
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        # K/V are projected to num_kv_heads streams, each of dim head_dim
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def _reshape_q(self, q: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.size()
        q = q.view(B, T, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        return q

    def _reshape_kv(self, proj: torch.Tensor) -> torch.Tensor:
        B, T, _ = proj.size()
        kv = proj.view(B, T, self.num_kv_heads, self.head_dim)
        kv = kv.transpose(1, 2)  # [B, Hkv, T, Dh]
        return kv

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        q = self._reshape_q(self.W_q(x))         # [B, H,   T, Dh]
        k = self._reshape_kv(self.W_k(x))        # [B, Hkv, T, Dh]
        v = self._reshape_kv(self.W_v(x))        # [B, Hkv, T, Dh]

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)  # [B, Hkv, T_total, Dh]
            v = torch.cat([kv_cache["v"], v], dim=2)  # [B, Hkv, T_total, Dh]

        new_cache = {"k": k, "v": v} if use_cache else None

        # Repeat each KV head group_size times to match num_heads
        # [B, Hkv, T_total, Dh] -> [B, H, T_total, Dh]
        k_tile = k.repeat_interleave(self.group_size, dim=1)
        v_tile = v.repeat_interleave(self.group_size, dim=1)

        Dh = self.head_dim
        att = torch.matmul(q, k_tile.transpose(-2, -1)) / math.sqrt(Dh)  # [B, H, T, T_total]

        if attn_mask is not None:
            att = att + attn_mask

        p = F.softmax(att, dim=-1)
        ctx = torch.matmul(p, v_tile)  # [B, H, T, Dh]

        ctx = ctx.transpose(1, 2).contiguous()  # [B, T, H, Dh]
        ctx = ctx.view(x.size(0), x.size(1), self.d_model)  # [B, T, d_model]

        out = self.W_o(ctx)
        return out, new_cache


# ---------------------------------------------------------------------------
# 4. Toy Multi-Head Latent Attention (MLA)
# ---------------------------------------------------------------------------

class MLA(nn.Module):
    """
    Toy Multi-Head Latent Attention.

    Idea:
      - Compute K/V for num_kv_heads heads (like GQA).
      - Compress them into a lower-dimensional latent space (latent_dim).
      - Store KV cache in the latent space (memory savings).
      - Optionally mix in latent space.
      - Map from latent back to head_dim for attention with Q heads.

    This is a conceptual simplification inspired by DeepSeek MLA,
    not an exact reproduction.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        latent_dim: int,
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
        self.latent_dim = latent_dim  # e.g., head_dim // 4

        # Q projection
        self.W_q = nn.Linear(d_model, d_model, bias=bias)

        # K/V: project to num_kv_heads * head_dim first
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)

        # Compress K/V per head_dim -> latent_dim
        self.W_k_to_latent = nn.Linear(self.head_dim, self.latent_dim, bias=bias)
        self.W_v_to_latent = nn.Linear(self.head_dim, self.latent_dim, bias=bias)

        # Optional latent mixing (identity here, but could be more complex)
        self.W_latent_mix_k = nn.Identity()
        self.W_latent_mix_v = nn.Identity()

        # Map from latent_dim back to head_dim
        self.W_k_from_latent = nn.Linear(self.latent_dim, self.head_dim, bias=bias)
        self.W_v_from_latent = nn.Linear(self.latent_dim, self.head_dim, bias=bias)

        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def _reshape_q(self, q: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.size()
        q = q.view(B, T, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        return q

    def _reshape_kv_heads(self, proj: torch.Tensor) -> torch.Tensor:
        B, T, _ = proj.size()
        kv = proj.view(B, T, self.num_kv_heads, self.head_dim)
        kv = kv.transpose(1, 2)  # [B, Hkv, T, Dh]
        return kv

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        # Q path
        q = self._reshape_q(self.W_q(x))           # [B, H, T, Dh]

        # K/V path (before latent compression)
        k = self._reshape_kv_heads(self.W_k(x))    # [B, Hkv, T, Dh]
        v = self._reshape_kv_heads(self.W_v(x))    # [B, Hkv, T, Dh]

        # Compress to latent_dim
        k_latent = self.W_k_to_latent(k)           # [B, Hkv, T, Dl]
        v_latent = self.W_v_to_latent(v)           # [B, Hkv, T, Dl]

        # Optional latent mixing
        k_latent = self.W_latent_mix_k(k_latent)
        v_latent = self.W_latent_mix_v(v_latent)

        # KV cache is stored in latent space
        if kv_cache is not None:
            k_latent = torch.cat([kv_cache["k_latent"], k_latent], dim=2)  # [B, Hkv, T_total, Dl]
            v_latent = torch.cat([kv_cache["v_latent"], v_latent], dim=2)  # [B, Hkv, T_total, Dl]

        new_cache = {"k_latent": k_latent, "v_latent": v_latent} if use_cache else None

        # Map latent back to head_dim for attention
        k_restored = self.W_k_from_latent(k_latent)  # [B, Hkv, T_total, Dh]
        v_restored = self.W_v_from_latent(v_latent)  # [B, Hkv, T_total, Dh]

        # Tile KV to match num_heads (grouped sharing)
        k_tile = k_restored.repeat_interleave(self.group_size, dim=1)  # [B, H, T_total, Dh]
        v_tile = v_restored.repeat_interleave(self.group_size, dim=1)  # [B, H, T_total, Dh]

        Dh = self.head_dim
        att = torch.matmul(q, k_tile.transpose(-2, -1)) / math.sqrt(Dh)  # [B, H, T, T_total]

        if attn_mask is not None:
            att = att + attn_mask

        p = F.softmax(att, dim=-1)
        ctx = torch.matmul(p, v_tile)  # [B, H, T, Dh]

        ctx = ctx.transpose(1, 2).contiguous()  # [B, T, H, Dh]
        ctx = ctx.view(x.size(0), x.size(1), self.d_model)  # [B, T, d_model]

        out = self.W_o(ctx)
        return out, new_cache


# ---------------------------------------------------------------------------
# 5. Simple causal mask helper (optional)
# ---------------------------------------------------------------------------

def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns an additive causal mask of shape [1, 1, T, T]
    where masked positions are -inf and unmasked are 0.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangle (strict)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
    return mask


# ---------------------------------------------------------------------------
# 6. Autoregressive cache growth + equivalence demo
# ---------------------------------------------------------------------------

def run_cache_demo_equivalence(module: nn.Module, name: str, d_model: int, device: torch.device):
    """
    Simulate autoregressive decoding on a FIXED sequence and check that:

      full_forward(x_full)  ≈  incremental_with_cache(x_full token by token)

    - At each step t, we feed x_full[:, t:t+1, :] (T_step = 1).
    - KV cache time dimension should grow by 1 each step.
    - We compare y_full[:, t] with the incremental output at that step.
    """
    torch.manual_seed(0)

    B = 1
    T = 6  # small for printing
    x_full = torch.randn(B, T, d_model, device=device)

    # Full forward with causal mask
    full_mask = causal_mask(T, device=device)
    y_full, _ = module(x_full, kv_cache=None, use_cache=False, attn_mask=full_mask)
    # y_full: [B, T, d_model]

    print(f"\n=== Cache equivalence demo: {name} ===")
    print("x_full shape:", tuple(x_full.shape))

    kv_cache = None
    y_inc_tokens = []

    for t in range(T):
        x_step = x_full[:, t:t+1, :]   # [B, 1, d_model]
        # No mask needed here: cache only contains past tokens
        y_step, kv_cache = module(x_step, kv_cache=kv_cache, use_cache=True, attn_mask=None)

        # y_step: [B, 1, d_model]
        y_inc_tokens.append(y_step)

        # Inspect cache shapes
        print(f"Step {t}:")
        print("  x_step shape:", tuple(x_step.shape))
        for k_name, tensor in kv_cache.items():
            print(f"  cache['{k_name}'].shape = {tuple(tensor.shape)}")

    # Concatenate incremental outputs over time
    y_inc = torch.cat(y_inc_tokens, dim=1)  # [B, T, d_model]

    # Compare full vs incremental (they should match up to numerical noise)
    max_abs_diff = (y_full - y_inc).abs().max().item()
    print(f"Max |y_full - y_inc| = {max_abs_diff:.6f}")


# ---------------------------------------------------------------------------
# 7. Smoke test in __main__
# ---------------------------------------------------------------------------

def smoke_test():
    torch.manual_seed(0)

    B = 2
    T = 16
    d_model = 512
    num_heads = 8
    num_kv_heads = 2
    latent_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x = torch.randn(B, T, d_model, device=device)
    mask = causal_mask(T, device=device)

    # -----------------------------
    # Single-shot tests with mask
    # -----------------------------
    print("\n=== Single-shot attention (with causal mask) ===")

    print("\n[MHA]")
    mha = MHA(d_model, num_heads).to(device)
    y_mha, cache_mha = mha(x, use_cache=True, attn_mask=mask)
    print("  Output shape:", y_mha.shape)
    print("  KV cache shapes:", cache_mha["k"].shape, cache_mha["v"].shape)

    print("\n[MQA]")
    mqa = MQA(d_model, num_heads).to(device)
    y_mqa, cache_mqa = mqa(x, use_cache=True, attn_mask=mask)
    print("  Output shape:", y_mqa.shape)
    print("  KV cache shapes:", cache_mqa["k"].shape, cache_mqa["v"].shape)

    print("\n[GQA]")
    gqa = GQA(d_model, num_heads, num_kv_heads=num_kv_heads).to(device)
    y_gqa, cache_gqa = gqa(x, use_cache=True, attn_mask=mask)
    print("  Output shape:", y_gqa.shape)
    print("  KV cache shapes:", cache_gqa["k"].shape, cache_gqa["v"].shape)

    print("\n[MLA (toy)]")
    mla = MLA(d_model, num_heads, num_kv_heads=num_kv_heads, latent_dim=latent_dim).to(device)
    y_mla, cache_mla = mla(x, use_cache=True, attn_mask=mask)
    print("  Output shape:", y_mla.shape)
    print("  Latent KV cache shapes:", cache_mla["k_latent"].shape, cache_mla["v_latent"].shape)

    # Sanity checks on output shapes
    assert y_mha.shape == (B, T, d_model)
    assert y_mqa.shape == (B, T, d_model)
    assert y_gqa.shape == (B, T, d_model)
    assert y_mla.shape == (B, T, d_model)

    print("\nSummary of KV cache head counts (single-shot):")
    print("  MHA KV heads:", cache_mha["k"].shape[1])
    print("  MQA KV heads:", cache_mqa["k"].shape[1])
    print("  GQA KV heads:", cache_gqa["k"].shape[1])
    print("  MLA latent KV heads:", cache_mla["k_latent"].shape[1])

    # -----------------------------
    # Autoregressive cache growth tests
    # -----------------------------
    run_cache_demo_equivalence(MHA(d_model, num_heads).to(device), "MHA", d_model, device)
    run_cache_demo_equivalence(MQA(d_model, num_heads).to(device), "MQA", d_model, device)
    run_cache_demo_equivalence(GQA(d_model, num_heads, num_kv_heads).to(device), "GQA", d_model, device)
    run_cache_demo_equivalence(MLA(d_model, num_heads, num_kv_heads, latent_dim).to(device), "MLA (toy)", d_model, device)

    print("\nDone.")


if __name__ == "__main__":
    smoke_test()
