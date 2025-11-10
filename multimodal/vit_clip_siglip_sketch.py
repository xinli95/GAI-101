# vit_clip_siglip_sketch.py
# Compact educational demo of ViT and CLIP/SigLIP losses.

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim=512, heads=8, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class MiniViT(nn.Module):
    def __init__(self, depth=4, dim=512, heads=8):
        super().__init__()
        self.patch = PatchEmbed(embed_dim=dim)
        self.blocks = nn.Sequential(*[ViTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, img):
        x = self.patch(img)
        x = self.blocks(x)
        x = self.norm(x)
        cls = x[:, 0]
        return F.normalize(self.proj(cls), dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, vocab=10000, dim=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, ids):
        x = self.emb(ids).mean(1)
        return F.normalize(self.proj(x), dim=-1)

def clip_loss(img_emb, txt_emb, temperature=0.07):
    sim = img_emb @ txt_emb.T / temperature
    labels = torch.arange(len(sim), device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

def siglip_loss(img_emb, txt_emb, temperature=0.07):
    sim = (img_emb @ txt_emb.T) / temperature
    target = torch.eye(sim.size(0), device=sim.device)
    return F.binary_cross_entropy_with_logits(sim, target)

def demo():
    B = 2
    model_v = MiniViT()
    model_t = TextEncoder()
    imgs = torch.randn(B, 3, 224, 224)
    txts = torch.randint(0, 10000, (B, 8))

    img_emb = model_v(imgs)
    txt_emb = model_t(txts)

    print("Image embedding shape:", img_emb.shape)
    print("CLIP loss:", float(clip_loss(img_emb, txt_emb)))
    print("SigLIP loss:", float(siglip_loss(img_emb, txt_emb)))

if __name__ == "__main__":
    demo()
