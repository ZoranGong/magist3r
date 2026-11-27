import torch


def get_multiview_scale(pts, valid):
    all_pts, all_nnz = [], 0
    for p, v in zip(pts, valid):
        p = p.clone()
        p[~v] = 0
        all_pts.append(p.flatten(1, 2))
        all_nnz += v.view(len(v), -1).sum(1)
    dis = torch.cat(all_pts, dim=1).norm(dim=-1)
    s = (dis.sum(dim=1) / (all_nnz + 1e-8)).clip(min=1e-8)
    while s.ndim < pts[0].ndim:
        s = s.unsqueeze(-1)
    return s


def top_ratio_mask(conf, ratio):
    flat = conf.reshape(-1, conf.shape[-2] * conf.shape[-1]).float()
    thr = torch.quantile(flat, 1.0 - ratio, dim=1, keepdim=True)
    return (flat >= thr).reshape(conf.shape)
