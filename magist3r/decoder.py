import torch

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def _attention(q, k, v, scale):
    if q.is_cuda and xops is not None:
        return xops.memory_efficient_attention(q.contiguous(), k.contiguous(), v.contiguous(), scale=scale)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    return (((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1) @ v).transpose(1, 2)


def _rope(rope, t, pos, n_skip):
    if n_skip == 0:
        return rope(t, pos)
    return torch.cat([t[:, :, :n_skip], rope(t[:, :, n_skip:].contiguous(), pos)], dim=2)


def self_attention(attn, x, pos, n_pose):
    B, N, C = x.shape
    H = attn.num_heads
    qkv = attn.qkv(x).reshape(B, N, 3, H, C // H).transpose(1, 3)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q, k = _rope(attn.rope, q, pos, n_pose), _rope(attn.rope, k, pos, n_pose)
    out = _attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn.scale)
    return attn.proj(out.reshape(B, N, C))


def cross_attention(blk, xs, ys, xpos, ypos, rel_ids, M, n_pose, mask_pose):
    ca = blk.cross_attn
    Vx, B, Nx, C = xs.shape
    Vy, _, Ny, _ = ys.shape
    H, K = ca.num_heads, C // ca.num_heads
    xs_n, ys_n = blk.norm2(xs), blk.norm_y(ys)
    if mask_pose:
        xs_n, q_skip = xs_n[:, :, n_pose:], 0
    else:
        q_skip = n_pose
    Nq = xs_n.shape[2]
    qs = _rope(ca.rope, ca.projq(xs_n).reshape(Vx * B, Nq, H, K).permute(0, 2, 1, 3), xpos, q_skip)
    ks = _rope(ca.rope, ca.projk(ys_n).reshape(Vy * B, Ny, H, K).permute(0, 2, 1, 3), ypos, n_pose)
    vs = ca.projv(ys_n).reshape(Vy, B, Ny, H, K)
    qs = qs.permute(0, 2, 1, 3).reshape(Vx, B, Nq, H, K)
    ks = ks.permute(0, 2, 1, 3).reshape(Vy, B, Ny, H, K)
    ks_sel = torch.index_select(ks, 0, rel_ids).reshape(Vx * M * B, Ny, H, K)
    vs_sel = torch.index_select(vs, 0, rel_ids).reshape(Vx * M * B, Ny, H, K)
    qs_rep = qs[:, None].expand(-1, M, -1, -1, -1, -1).reshape(Vx * M * B, Nq, H, K)
    out = _attention(qs_rep, ks_sel, vs_sel, ca.scale).reshape(Vx, M, B, Nq, C)
    return ca.proj(out)


def block_forward(blk, xs, ys, xpos, ypos, rel_ids, M, n_pose, mask_pose):
    Vx, B, Nx, C = xs.shape
    x = xs.reshape(Vx * B, Nx, C)
    x = x + self_attention(blk.attn, blk.norm1(x), xpos, n_pose)
    merged = cross_attention(blk, x.reshape(Vx, B, Nx, C), ys, xpos, ypos, rel_ids, M, n_pose, mask_pose)
    merged = merged.max(dim=1).values
    if merged.shape[2] != Nx:
        merged = torch.cat([merged.new_zeros(Vx, B, Nx - merged.shape[2], C), merged], dim=2)
    x = x + merged.reshape(Vx * B, Nx, C)
    x = x + blk.mlp(blk.norm3(x))
    return x.reshape(Vx, B, Nx, C)


def _add_geometry(x, geo, n_pose):
    return x + torch.cat([geo.new_zeros(*geo.shape[:2], n_pose, geo.shape[-1]), geo], dim=2)


def magma_decode(model, ref_x, src_x, ref_geo, src_geo, ref_pos, src_pos, n_pose):
    R, S, B = ref_x.shape[0], src_x.shape[0], ref_x.shape[1]
    device = ref_x.device
    src_rel = torch.arange(R, device=device, dtype=torch.long)[None].expand(S, -1).reshape(-1)
    ref_rel = torch.arange(S, device=device, dtype=torch.long)[None].expand(R, -1).reshape(-1)
    refs, srcs = ref_x, src_x
    for i in range(model.dec_depth):
        ref_in, src_in = _add_geometry(refs, ref_geo, n_pose), _add_geometry(srcs, src_geo, n_pose)
        refs = block_forward(model.mv_dec_blocks1[i], ref_in, src_in, ref_pos, src_pos, ref_rel, S, n_pose, True)
        srcs = block_forward(model.mv_dec_blocks2[i], src_in, ref_in, src_pos, ref_pos, src_rel, R, n_pose, False)
    refs, srcs = model.dec_norm(refs), model.dec_norm(srcs)
    return refs.reshape(R * B, *refs.shape[2:]), srcs.reshape(S * B, *srcs.shape[2:])
