import torch
import torch.nn as nn

from . import NUM_POSE_TOKENS, L2W_GRID, L2W_RES
from .decoder import magma_decode
from .fusion import MagmaFusion
from .layers import DecoderBlock, LinearPts3d, PatchEmbed, PositionGetter
from .pos_embed import RoPE2D
from .pose_head import MagmaPoseHead

PATCH = 16
ENC_DIM = 1024
DEC_DIM = 768
DEC_DEPTH = 12
DEC_HEADS = 12


class MagmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = RoPE2D()
        self.patch_embed = PatchEmbed(PATCH, ENC_DIM)
        self.enc_norm = nn.LayerNorm(ENC_DIM, eps=1e-6)
        self.decoder_embed = nn.Linear(ENC_DIM, DEC_DIM)
        self.dec_depth = DEC_DEPTH
        self.mv_dec_blocks1 = nn.ModuleList([DecoderBlock(DEC_DIM, DEC_HEADS, self.rope) for _ in range(DEC_DEPTH)])
        self.mv_dec_blocks2 = nn.ModuleList([DecoderBlock(DEC_DIM, DEC_HEADS, self.rope) for _ in range(DEC_DEPTH)])
        self.dec_norm = nn.LayerNorm(DEC_DIM, eps=1e-6)
        self.downstream_head1 = LinearPts3d(DEC_DIM, PATCH)
        self.downstream_head2 = LinearPts3d(DEC_DIM, PATCH)
        self.void_pe_token = nn.Parameter(torch.zeros(1, 1, DEC_DIM))
        self.ponit_embedder = nn.Conv2d(3, DEC_DIM, kernel_size=PATCH, stride=PATCH)
        self.fusion = MagmaFusion(DEC_DIM, DEC_HEADS)
        self.pose_head = MagmaPoseHead(DEC_DIM)
        self.patch_positions = PositionGetter()

    def embed_points(self, pts):
        pos = pts.permute(0, 3, 1, 2)
        return self.ponit_embedder(pos).permute(0, 2, 3, 1).reshape(pos.shape[0], -1, DEC_DIM)

    def forward(self, views, ref_ids):
        V, n = len(views), NUM_POSE_TOKENS
        src_ids = [i for i in range(V) if i not in ref_ids]
        geo = [self.embed_points(v['pts3d_world'] if i in ref_ids else v['pts3d_cam']) for i, v in enumerate(views)]
        B = views[0]['vggt_tokens'].shape[0]
        vis_e, geo_e, pose_g = self.fusion(torch.cat([v['vggt_tokens'] for v in views]), torch.cat(geo),
                                           torch.cat([v['salad_tokens'] for v in views]),
                                           torch.cat([v['pose_tokens'] for v in views]))
        D = vis_e.shape[-1]
        geo_e = geo_e.view(V, B, -1, D)
        x = torch.cat([pose_g.view(V, B, n, D), vis_e.view(V, B, -1, D)], dim=2)
        ref_idx = torch.tensor(ref_ids, device=x.device)
        src_idx = torch.tensor(src_ids, device=x.device)
        R, S = len(ref_ids), len(src_ids)
        dec_ref, dec_src = magma_decode(self, x[ref_idx], x[src_idx], geo_e[ref_idx], geo_e[src_idx],
                                        self.patch_positions(R * B, L2W_GRID, L2W_GRID, x.device),
                                        self.patch_positions(S * B, L2W_GRID, L2W_GRID, x.device), n)
        results = [dict() for _ in range(V)]
        with torch.autocast(device_type='cuda', enabled=False):
            res_ref = self.downstream_head1(dec_ref[:, n:].float(), L2W_RES)
            res_src = self.downstream_head2(dec_src[:, n:].float(), L2W_RES)
            groups = torch.cat([dec_ref[:, :n].reshape(R, B, n, D), dec_src[:, :n].reshape(S, B, n, D)], dim=0)
            pose_enc = self.pose_head(groups.permute(1, 0, 2, 3))
        for k, vid in enumerate(ref_ids):
            results[vid]['pts3d'] = res_ref['pts3d'][k * B:(k + 1) * B]
            results[vid]['conf'] = res_ref['conf'][k * B:(k + 1) * B]
        for k, vid in enumerate(src_ids):
            results[vid]['pts3d_in_other_view'] = res_src['pts3d'][k * B:(k + 1) * B]
            results[vid]['conf'] = res_src['conf'][k * B:(k + 1) * B]
        for k, vid in enumerate(ref_ids + src_ids):
            results[vid]['pose_enc'] = pose_enc[:, k]
        return results


def load_model(path, device):
    model = MagmaModel()
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith('enc_blocks')}, strict=True)
    return model.to(device).eval()
