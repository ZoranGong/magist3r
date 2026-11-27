import torch

HYPOTHESES = 512
SAMPLE_SIZE = 4
SCORE_POINTS = 20000
INLIER_FRAC = 0.02


def weighted_umeyama(src, dst, w, eps=1e-9):
    src, dst, w = src.double(), dst.double(), w.double().clamp(min=0)
    w = w / w.sum().clamp(min=eps)
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (w[:, None] * xd).T @ xs + eps * torch.eye(3, dtype=torch.float64, device=src.device)
    U, D, Vt = torch.linalg.svd(cov)
    S = torch.ones(3, dtype=torch.float64, device=src.device)
    S[2] = torch.sign(torch.det(U) * torch.det(Vt))
    R = U @ torch.diag(S) @ Vt
    s = (D * S).sum() / (w * (xs ** 2).sum(1)).sum().clamp(min=eps)
    return s, R, mu_d - s * R @ mu_s


def apply_sim3(s, R, t, pts):
    return (s * pts.double() @ R.T + t).to(pts.dtype)


def _residuals(s, R, t, src, dst):
    return (s * src.double() @ R.T + t - dst.double()).norm(dim=-1)


def irls_sim3(src, dst):
    base = torch.ones(len(src), dtype=torch.float64, device=src.device)
    w = base
    for _ in range(3):
        s, R, t = weighted_umeyama(src, dst, w)
        r = _residuals(s, R, t, src, dst)
        w = base / (1.0 + (r / (2.0 * r.median()).clamp(min=1e-6)) ** 2)
    s, R, t = weighted_umeyama(src, dst, w)
    return s, R, t, _residuals(s, R, t, src, dst)


@torch.no_grad()
def ransac_sim3(src, dst, conf):
    dev = src.device
    src, dst, conf = src.double(), dst.double(), conf.double()
    if src.shape[0] > SCORE_POINTS:
        sel = torch.randperm(src.shape[0], device=dev)[:SCORE_POINTS]
        ss, dd, cc = src[sel], dst[sel], conf[sel]
    else:
        ss, dd, cc = src, dst, conf
    idx = torch.multinomial((cc / cc.sum()).float(), HYPOTHESES * SAMPLE_SIZE, replacement=True)
    idx = idx.view(HYPOTHESES, SAMPLE_SIZE)
    A, B = ss[idx], dd[idx]
    mA, mB = A.mean(1, keepdim=True), B.mean(1, keepdim=True)
    XA, XB = A - mA, B - mB
    U, D, Vt = torch.linalg.svd(XB.transpose(1, 2) @ XA / SAMPLE_SIZE)
    S = torch.ones(HYPOTHESES, 3, dtype=torch.float64, device=dev)
    S[:, 2] = torch.sign(torch.det(U) * torch.det(Vt))
    R = U @ torch.diag_embed(S) @ Vt
    s = (D * S).sum(1) / ((XA ** 2).sum((1, 2)) / SAMPLE_SIZE).clamp(min=1e-12)
    t = mB[:, 0] - s[:, None] * (R @ mA[:, 0, :, None])[..., 0]
    thr = INLIER_FRAC * (dd - dd.mean(0)).norm(dim=1).median()
    pred = s[:, None, None] * torch.einsum('hij,nj->hni', R, ss) + t[:, None, :]
    score = (((pred - dd[None]).norm(dim=-1) < thr).double() * cc[None]).sum(1)
    score[~(torch.isfinite(s) & (s > 0))] = -1
    best = int(score.argmax())
    mask = _residuals(s[best], R[best], t[best], src, dst) < thr
    if mask.sum() < 3 * SAMPLE_SIZE:
        mask = torch.ones_like(mask)
    s_f, R_f, t_f = weighted_umeyama(src[mask], dst[mask], conf[mask])
    inl = _residuals(s_f, R_f, t_f, src, dst) < thr
    return dict(s=s_f, R=R_f, t=t_f, inliers=inl, inlier_ratio=float(inl.double().mean()))
