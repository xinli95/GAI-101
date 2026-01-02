import torch
import torch.nn as nn

# architecture diagram: https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt2-to-llama2-llama3.webp?1



class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.shift + self.scale * norm_x
    
class RMSNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.sqrt(mean + self.eps)
        return (norm_x * self.weight).to(dtype=x.dtype)
    

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
    
def precompute_rope_params(head_dim, theta_base=10_000, ctx_len=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    #Compute inverse freq
    inv_feq = 1/ (theta_base ** (torch.arange(head_dim, 2) / head_dim))
    pos = torch.arange(ctx_len)

    angles = pos[:, None] * inv_feq[None, :]

    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, sin, cos):
    # x : (b x h x l x d)

    batch_size, num_heads, seq_len, head_dim = x.shape

    # half split
    x1 = x[..., :head_dim//2]
    x2 = x[..., head_dim//2:]

    # adjust shape
    cos = cos[:seq_len,:].unsqueeze(0).unsqueeze(0) # 1 x 1 x seq_len x head_dim
    sin = sin[:seq_len,:].unsqueeze(0).unsqueeze(0) # 1 x 1 x seq_len x head_dim

    # apply rotation
    rotated_x = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated_x * sin

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, ctx_len, num_heads):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.ctx_len = ctx_len
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1))
        cos, sin = precompute_rope_params(head_dim=self.head_dim,ctx_len=ctx_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        b, seq_len, d_in = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(b, seq_len, self.num_heads, self.head_dim)
        k = k.view(b, seq_len, self.num_heads, self.head_dim)
        v = v.view(b, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = compute_rope(k, self.sin, self.cos)
        q = compute_rope(q, self.sin, self.cos)

        attn = q @ k.transpose(2, 3)
        mask = self.mask.bool()[:seq_len, :seq_len]
        attn.masked_fill_(mask, -torch.inf)
        attn = torch.softmax(attn / k.shape[-1] ** 0.5, dim=-1)

        ctx = (attn @ v).transpose(1, 2)
        ctx = ctx.reshape(b, seq_len, self.d_out)
        ctx = self.out_proj(ctx)

        return ctx

class GroupQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_heads):
        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_out // num_heads
        self.group_size = self.num_heads // self.num_kv_heads

        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_in, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
    
    def forward(self, x, mask=None, cos=None, sin=None):
        b, s, d_in = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_kv_heads, self.head_dim)
        v = v.view(b, s, self.num_kv_heads, self.head_dim)

        # shuffle
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # apply rope
        if cos is not None:
            k = compute_rope(k, cos, sin)
            q = compute_rope(q, cos, sin)
        
        # expand kv to match num_heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        att = (q @ k.transepo(2, 3)) / self.head_dim ** 0.5

        if mask is None:
            mask = torch.triu(torch.ones(s, s, device=x.device), diagonal=1)
        att.mask_fill_(mask, -torch.inf)
        att = torch.softmax(att, dim=-1)
        ctx = (att @ v).transpose(1, 2)
        ctx = ctx.reshape(b, s, -1)
        ctx = self.out_proj(ctx)
        return ctx
        


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg.emb_dim,
        num_heads = cfg.num_heads,
        num_kv_heads = cfg.num_kv_heads,
        hidden_dim = cfg.hidden_dim

        self.norm1 = RMSNorm(emb_dim)
        self.norm2 = RMSNorm(emb_dim)
        self.att = GroupQueryAttention(emb_dim, emb_dim, num_heads, num_kv_heads)
        self.ff = FeedForward(emb_dim, hidden_dim)

    def forward(self, x, mask=None, cos=None, sin=None):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x= self.ff(x)
        x += shortcut

        return x
    
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_size)
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        cos, sin = precompute_rope_params(cfg.head_dim / cfg.num_heads, cfg.theta_base, cfg.ctx_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.cfg = cfg

    def forward(self, input_idx):
        token_embeds = self.tok_emb(input_idx)
        x = token_embeds
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

        for block in self.blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

