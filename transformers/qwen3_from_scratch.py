import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg.emd_dim
        hidden_dim = cfg.hidden_dim

        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.fc1(x) * self.silu(self.fc2(x))
        return self.fc3(x)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
        self.eps = 1e-5
    
    def forward(self, x):

        norm_x = x / torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        norm_x = norm_x * self.weight
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x


def compute_rope_params(head_dim, theta_base, ctx_len):

    p = torch.arange(ctx_len)
    inv_freq = 1 / theta_base ** (torch.arange(head_dim // 2) / head_dim)

    angles = p.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=-1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def compute_rope(x, cos, sin):
    b, h, s, hd = x.shape
    x1 = x[..., :hd//2]
    x2 = x[..., hd//2:]
    cos = cos[:s, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:s, :].unsqueeze(0).unsqueeze(0)
    rotated_x = torch.cat([-x2, x1], dim=-1)

    return x * cos + rotated_x * sin


class GroupQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.emb_dim = cfg.emd_dim
        self.head_dim = self.emb_dim // self.num_heads
        self.group_size = self.num_heads // self.num_kv_heads
        self.qk_norm = cfg.qk_norm

        self.W_q = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W_k = nn.Linear(self.emb_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(self.emb_dim,self.num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None
    
    def forward(self, x, mask, cos, sin):
        b, seq_len, d = x.shape
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape to heads
        q = q.view(b, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(b, seq_len, self.num_kv_heads, -1).transpose(1, 2)
        v = v.view(b, seq_len, self.num_kv_heads, -1).transpose(1, 2)

        # optional normalization
        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)
        
        # apply rope
        q = compute_rope(x, cos, sin)
        k = compute_rope(x, cos, sin)

        # expand kv
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # attention
        att = (q @ k.transpose(2, 3)) / torch.sqrt(self.head_dim)
        att.masked_fill(mask, -torch.inf)
        att = torch.softmax(att, dim=-1)

        ctx = (att @ v).transpose(1, 2).reshape(b, seq_len, d)
        return self.output_proj(ctx)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupQueryAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(emb_dim=cfg.emb_dim)
        self.norm2 = RMSNorm(emb_dim=cfg.emb_dim)

    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.blocks = nn.Sequential(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.emd_dim)
        self.output_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        cos, sin = compute_rope_params(cfg.head_dim, cfg.theta_base, cfg.ctx_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.cfg = cfg
    
    def forward(self, input_ind):
        token_embed = self.token_embed(input_ind)
        x = token_embed
        
        # generate mask based on current seq_len
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

        for block in self.blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits