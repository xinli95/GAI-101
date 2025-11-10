# blip_qformer_sketch.py
# A compact PyTorch sketch of the BLIP-2 Q-Former idea:
# - Learnable query tokens
# - Self-attention on queries
# - Cross-attention from queries to frozen vision tokens
# - Projection to LLM embedding space
#
# This is a teaching scaffold, not a drop-in training script.
# Replace the toy loss with ITM/LM objectives for real training.

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Transformer building blocks
# -------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_mult=4, p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_mult * dim)
        self.fc2 = nn.Linear(hidden_mult * dim, dim)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention(nn.Module):
    """Self-attention over the M query tokens."""
    def __init__(self, dim, heads=12, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)

    def forward(self, q_tokens):  # (B, M, D)
        x = self.ln(q_tokens)
        out, _ = self.attn(x, x, x, need_weights=False)
        return q_tokens + self.drop(out)

class CrossAttention(nn.Module):
    """Cross-attention: queries (M) attend to vision tokens (N)."""
    def __init__(self, q_dim, kv_dim, heads=12, p=0.1):
        super().__init__()
        self.q_ln  = nn.LayerNorm(q_dim)
        self.kv_ln = nn.LayerNorm(kv_dim)
        # MultiheadAttention expects q,k,v to have same embed_dim => project kv to q_dim
        self.kv_proj = nn.Linear(kv_dim, q_dim)
        self.attn = nn.MultiheadAttention(embed_dim=q_dim, num_heads=heads, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)

    def forward(self, q_tokens, v_tokens):  # q: (B, M, Dq), v: (B, N, Dv)
        q = self.q_ln(q_tokens)
        kv = self.kv_ln(v_tokens)
        kv = self.kv_proj(kv)               # (B, N, Dq)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return q_tokens + self.drop(out)

class QFormerBlock(nn.Module):
    """(Self-Attn on queries) -> (Cross-Attn to vision) -> (MLP)."""
    def __init__(self, q_dim, kv_dim, heads=12, mlp_mult=4, p=0.1):
        super().__init__()
        self.self_attn = SelfAttention(q_dim, heads=heads, p=p)
        self.cross_attn = CrossAttention(q_dim, kv_dim, heads=heads, p=p)
        self.ln = nn.LayerNorm(q_dim)
        self.ff = FeedForward(q_dim, hidden_mult=mlp_mult, p=p)

    def forward(self, q_tokens, v_tokens):
        q_tokens = self.self_attn(q_tokens)
        q_tokens = self.cross_attn(q_tokens, v_tokens)
        q_tokens = q_tokens + self.ff(self.ln(q_tokens))
        return q_tokens

# -------------------------------
# Q-Former module
# -------------------------------

class QFormer(nn.Module):
    def __init__(self, num_queries=32, q_dim=768, kv_dim=768, depth=4, heads=12, p=0.1):
        super().__init__()
        # Learnable query tokens (parameters, not derived from inputs)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, q_dim) / (q_dim ** 0.5))

        self.blocks = nn.ModuleList([
            QFormerBlock(q_dim=q_dim, kv_dim=kv_dim, heads=heads, p=p)
            for _ in range(depth)
        ])
        self.out_ln = nn.LayerNorm(q_dim)

    def forward(self, vision_tokens):
        """
        vision_tokens: (B, N, Dv) from a frozen ViT (patch embeddings + optional CLS)
        returns: summarized tokens (B, M, Dq)
        """
        B = vision_tokens.size(0)
        # Tile the learnable query bank for the batch
        q = self.query_tokens.expand(B, -1, -1).contiguous()  # (B, M, Dq)

        for blk in self.blocks:
            q = blk(q, vision_tokens)

        return self.out_ln(q)  # (B, M, Dq)

# -------------------------------
# Demo: wire Q-Former with frozen vision tokens + projector to LLM space
# -------------------------------

def demo_run():
    # Hypers (illustrative)
    B, N, Dv = 2, 196, 768   # batch, #vision tokens, vision dim (e.g., CLIP ViT-L/14)
    M, Dq = 32, 768          # #query tokens, query dim
    LLM_DIM = 4096           # LLM embedding size (e.g., a GPT/T5 variant)

    # (A) Frozen vision encoder output (stand-in: random tensor)
    with torch.no_grad():
        vision_tokens = torch.randn(B, N, Dv)  # Real use: pass images through a frozen ViT

    # (B) Q-Former (trainable)
    qformer = QFormer(num_queries=M, q_dim=Dq, kv_dim=Dv, depth=4, heads=12, p=0.1)

    # (C) Projection to LLM token space (trainable)
    proj_to_llm = nn.Linear(Dq, LLM_DIM)

    # Forward pass
    q_tokens = qformer(vision_tokens)          # (B, M, Dq)
    llm_visual_tokens = proj_to_llm(q_tokens)  # (B, M, LLM_DIM)

    print("llm_visual_tokens:", llm_visual_tokens.shape)

    # -------------------
    # Toy optimization step (replace with ITM/LM losses)
    # -------------------
    targets = torch.randn(B, M, LLM_DIM)
    opt = torch.optim.AdamW(
        list(qformer.parameters()) + list(proj_to_llm.parameters()),
        lr=1e-4, weight_decay=0.01
    )
    loss = F.mse_loss(llm_visual_tokens, targets)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print("loss:", float(loss.item()))

if __name__ == '__main__':
    demo_run()
