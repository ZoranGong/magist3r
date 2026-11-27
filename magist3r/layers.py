# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, rope):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = rope


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, rope):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.projq = nn.Linear(dim, dim)
        self.projk = nn.Linear(dim, dim)
        self.projv = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = rope


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, rope):
        super().__init__()
        norm = lambda: nn.LayerNorm(dim, eps=1e-6)
        self.norm1, self.norm2, self.norm3, self.norm_y = norm(), norm(), norm(), norm()
        self.attn = Attention(dim, num_heads, rope)
        self.cross_attn = CrossAttention(dim, num_heads, rope)
        self.mlp = Mlp(dim, dim * 4)


class PositionGetter:
    def __init__(self):
        self.cache = {}

    def __call__(self, b, h, w, device):
        if (h, w) not in self.cache:
            self.cache[h, w] = torch.cartesian_prod(torch.arange(h, device=device), torch.arange(w, device=device))
        return self.cache[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)


class LinearPts3d(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(dim, 4 * patch_size ** 2)

    def forward(self, tokens, hw):
        B = tokens.shape[0]
        feat = self.proj(tokens).transpose(-1, -2).view(B, -1, hw // self.patch_size, hw // self.patch_size)
        fmap = F.pixel_shuffle(feat, self.patch_size).permute(0, 2, 3, 1)
        xyz = fmap[..., :3]
        d = xyz.norm(dim=-1, keepdim=True)
        return dict(pts3d=xyz / d.clip(min=1e-8) * torch.expm1(d), conf=1 + fmap[..., 3].exp())
