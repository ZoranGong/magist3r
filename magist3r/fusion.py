import torch
import torch.nn as nn

from . import NUM_POSE_TOKENS

VGGT_DIM = 2048
SALAD_DIM = 768


class TokenAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class ResidualAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


class SpatialCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, visual, spatial):
        kv = self.norm_kv(spatial)
        out, _ = self.attn(self.norm_q(visual), kv, kv, need_weights=False)
        return out + visual


class MagmaFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.visual_adapter = TokenAdapter(VGGT_DIM, dim)
        self.spatial_adapter = TokenAdapter(SALAD_DIM, dim)
        self.cross_attn = SpatialCrossAttention(dim, num_heads)
        self.geometry_adapter = ResidualAdapter(dim)
        self.pose_adapter = TokenAdapter(VGGT_DIM, dim)
        self.pose_pos_embed = nn.Parameter(torch.zeros(1, NUM_POSE_TOKENS, dim))

    def forward(self, visual_tokens, geometry_tokens, spatial_tokens, pose_tokens):
        vis = self.cross_attn(self.visual_adapter(visual_tokens), self.spatial_adapter(spatial_tokens))
        return vis, self.geometry_adapter(geometry_tokens), self.pose_adapter(pose_tokens) + self.pose_pos_embed
