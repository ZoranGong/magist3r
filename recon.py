#!/usr/bin/env python3
import argparse
import os
import warnings
import re
from os.path import join

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
import trimesh

warnings.filterwarnings('ignore')

from magist3r import L2W_RES, ROOT_DIR
from magist3r.backbones import Backbones
from magist3r.keyframes import make_windows, KeyframeDatabase
from magist3r.model import load_model
from magist3r.pgo import optimize_pose_graph, to_sim3, from_sim3, correction
from magist3r.sim3 import ransac_sim3, apply_sim3, irls_sim3, weighted_umeyama
from magist3r.utils import get_multiview_scale, top_ratio_mask
from vggt_omega.utils.pose_enc import encoding_to_camera

CFG = dict(
    vggt_resolution=416, depth_conf_thres=1.5, win_len=10, win_stride=7, keyframe_ids=(0, 4),
    overlap_thres=0.90, overlap_min_gap=8, overlap_max_gap=40, overlap_coarse_step=4,
    num_refs=6, ref_keep_ratio=0.9, bootstrap_windows=2, l2w_conf_thres=3.0, salad_thres=0.70,
    min_inlier_ratio=0.25, min_inlier_conf=3.0, anchor_rot_tol=1.5, anchor_scale_tol=0.02,
    loop_min_gap=60, loop_rel_weight=0.02, ply_keep_ratio=0.95, num_points=2000000, seed=42)
WEIGHTS = dict(model=join(ROOT_DIR, 'weights', 'magist3r.pth'),
               vggt=join(ROOT_DIR, 'weights', 'vggt_omega_1b_416_reproduce.pt'),
               salad=join(ROOT_DIR, 'weights', 'dino_salad_2048_64.ckpt'))


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--gpu_id', type=int, default=0)
    return p


def list_images(img_dir):
    names = [n for n in os.listdir(img_dir) if n.endswith('.jpg')]

    def key(n):
        nums = re.findall(r'\d+', n)
        return (int(nums[-1]) if nums else -1, n)
    return [join(img_dir, n) for n in sorted(names, key=key)]


def load_frame(path, hr_res):
    img = PIL.Image.open(path).convert('RGB')
    W, H = img.size
    s = min(W, H)
    left, top = (W - s) // 2, (H - s) // 2
    img = img.crop((left, top, left + s, top + s))
    hr = img.resize((hr_res, hr_res), PIL.Image.Resampling.BICUBIC)
    lr = img.resize((L2W_RES, L2W_RES), PIL.Image.Resampling.LANCZOS)
    return dict(img_hr=TF.to_tensor(hr), rgb=np.asarray(lr, dtype=np.uint8))


def transform_pts(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def pose_from_encoding(pose_enc):
    extr, _ = encoding_to_camera(pose_enc.reshape(1, 1, 9).float(), (224, 224))
    w2c = torch.eye(4, device=extr.device, dtype=extr.dtype)
    w2c[:3] = extr[0, 0]
    return torch.linalg.inv(w2c)


@torch.no_grad()
def covisibility(backbones, img_a, imgs_b, conf_thres):
    B = imgs_b.shape[0]
    f = backbones(torch.stack([img_a.expand(B, -1, -1, -1), imgs_b], dim=1), need_salad=False)
    _, intr = encoding_to_camera(f['pose_enc'].float(), (L2W_RES, L2W_RES))
    out = []
    for b in range(B):
        P = f['world_points'][b, 0][f['depth_conf'][b, 0] > conf_thres].double()
        if P.shape[0] < 50:
            out.append(0.0)
            continue
        w2c = torch.linalg.inv(f['vggt_c2w'][b, 1].double())
        pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        uv = pc @ intr[b, 1].double().T
        u, v = uv[:, 0] / z.clamp(min=1e-6), uv[:, 1] / z.clamp(min=1e-6)
        inside = (z > 0) & (u >= 0) & (u <= L2W_RES - 1) & (v >= 0) & (v <= L2W_RES - 1)
        d = torch.zeros_like(z)
        d[inside] = f['cam_points'][b, 1][v[inside].round().long(), u[inside].round().long(), 2].double()
        out.append(float((inside & (d > 0) & ((z - d).abs() < 0.1 * d)).double().mean()))
    return out


def select_frames(paths, backbones, device):
    n, imgs = len(paths), {}

    def img(i):
        if i not in imgs:
            imgs[i] = load_frame(paths[i], CFG['vggt_resolution'])['img_hr']
        return imgs[i]

    def covis(f, ks):
        return covisibility(backbones, img(f).to(device), torch.stack([img(f + k) for k in ks]).to(device),
                            CFG['depth_conf_thres'])

    sel = [0]
    while sel[-1] < n - 1:
        f = sel[-1]
        for i in [i for i in imgs if i < f]:
            del imgs[i]
        max_k = min(CFG['overlap_max_gap'], n - 1 - f)
        cs = CFG['overlap_coarse_step']
        coarse = list(range(cs, max_k + 1, cs)) or [max_k]
        cov = covis(f, coarse)
        fail = next((i for i, c in enumerate(cov) if c < CFG['overlap_thres']), None)
        if fail is None:
            lo, hi = coarse[-1], max_k + 1
        else:
            lo, hi = (coarse[fail - 1] if fail > 0 else 0), coarse[fail]
        if hi - 1 < CFG['overlap_min_gap']:
            lo, hi = hi - 1, hi
        k, fine = lo, list(range(lo + 1, hi))
        if fine:
            for kk, c in zip(fine, covis(f, fine)):
                if c < CFG['overlap_thres']:
                    break
                k = kk
        sel.append(f + max(k, min(CFG['overlap_min_gap'], n - 1 - f), 1))
    return sel


def retrieve(pool, query_desc, k):
    ids = sorted(pool)
    descs = torch.stack([pool[i]['desc'] for i in ids]).to(query_desc)
    sim = (query_desc @ descs.T).max(dim=0).values.float().cpu().numpy()
    low = np.array([bool(pool[i].get('low_conf', False)) for i in ids])
    order = np.lexsort((np.arange(len(ids)), -sim, low))[:min(k, len(ids))]
    return [ids[j] for j in order], float(sim.max())


@torch.no_grad()
def main(args):
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda')
    torch.manual_seed(CFG['seed'])
    rng = np.random.default_rng(CFG['seed'])
    os.makedirs(args.out_dir, exist_ok=True)
    paths = list_images(args.img_dir)
    assert paths, f'no images in {args.img_dir}'
    backbones = Backbones(WEIGHTS['vggt'], WEIGHTS['salad']).to(device)
    selected = select_frames(paths, backbones, device)
    windows = [[selected[k] for k in win] for win in make_windows(len(selected), CFG['win_len'], CFG['win_stride'])]
    model = load_model(WEIGHTS['model'], device)

    db, frames, cache, prev_pool = KeyframeDatabase(), {}, {}, {}
    nodes, anchors, loops, frame_rgb = {}, {}, {}, {}
    for w, win in enumerate(windows):
        for f in [f for f in cache if f < win[0]]:
            del cache[f]
        for f in win:
            if f not in cache:
                cache[f] = load_frame(paths[f], CFG['vggt_resolution'])
            frame_rgb.setdefault(f, cache[f]['rgb'])
        V = len(win)
        anchor = None
        if w > 0 and windows[w - 1][-1] in prev_pool:
            af = windows[w - 1][-1]
            anchor = dict(entry=prev_pool[af], img=load_frame(paths[af], CFG['vggt_resolution'])['img_hr'])
        feats = backbones(torch.stack([cache[f]['img_hr'] for f in win] + ([anchor['img']] if anchor else []))[None]
                          .to(device))
        if anchor:
            anchor['world_points'] = feats['world_points'][0, V]
            anchor['conf'] = feats['depth_conf'][0, V]
            feats = {k: (v[:, :V] if torch.is_tensor(v) and v.ndim > 1 and v.shape[1] == V + 1 else v)
                     for k, v in feats.items()}
        conf_masks = feats['depth_conf'][0] > CFG['depth_conf_thres']

        def frame_entry(j, pts, conf, c2w, low_conf=False):
            return dict(low_conf=low_conf, window=w, vggt_tokens=feats['vggt_tokens'][0, j].half().cpu(),
                        pose_tokens=feats['pose_tokens'][0, j].half().cpu(),
                        salad_tokens=feats['salad_tokens'][0, j].half().cpu(),
                        desc=feats['global_descriptor'][0, j].float().cpu(),
                        pts=pts.cpu(), c2w=c2w.cpu(), valid=top_ratio_mask(conf, CFG['ref_keep_ratio']).cpu())

        new_pool = {}
        if w == 0:
            to_global = torch.linalg.inv(feats['vggt_c2w'][0, 0].double())
            nodes[0] = (torch.tensor(1.0, dtype=torch.float64), to_global[:3, :3].cpu(), to_global[:3, 3].cpu())
            for j, f in enumerate(win):
                pts = transform_pts(to_global, feats['world_points'][0, j].double()).float()
                c2w = torch.eye(4, dtype=torch.float64, device=device) if j == 0 else \
                    to_global @ feats['vggt_c2w'][0, j].double()
                conf = feats['depth_conf'][0, j].float()
                frames[f] = dict(c2w=c2w.cpu().numpy(), pts=pts.cpu().numpy(), conf=conf.cpu().numpy(),
                                 valid=conf_masks[j].cpu().numpy(), score=np.inf, window=w)
                entry = frame_entry(j, pts, conf, c2w)
                new_pool[f] = dict(entry, score=np.inf)
                if j in CFG['keyframe_ids']:
                    db.add_or_update(f, entry, score=np.inf)
            prev_pool = new_pool
            continue

        pool = dict(db.entries)
        if w <= CFG['bootstrap_windows']:
            for f, e in prev_pool.items():
                pool.setdefault(f, e)
        ref_frames, top1 = retrieve(pool, feats['global_descriptor'][0], CFG['num_refs'])
        if top1 < CFG['salad_thres']:
            pool = dict(db.entries)
            for f, e in prev_pool.items():
                pool.setdefault(f, e)
            ref_frames, _ = retrieve(pool, feats['global_descriptor'][0], CFG['num_refs'])
        P0 = pool[ref_frames[0]]['c2w'].to(device).double()
        to_primary = torch.linalg.inv(P0)
        ref_local = [transform_pts(to_primary, pool[f]['pts'].to(device).double()).float()[None] for f in ref_frames]
        ref_valid = [pool[f]['valid'].to(device)[None] for f in ref_frames]
        s = get_multiview_scale(ref_local, ref_valid)
        views = []
        for f, pts, valid in zip(ref_frames, ref_local, ref_valid):
            e = pool[f]
            views.append(dict(pts3d_world=pts / s * valid[..., None].float(),
                              vggt_tokens=e['vggt_tokens'].to(device).float()[None],
                              pose_tokens=e['pose_tokens'].to(device).float()[None],
                              salad_tokens=e['salad_tokens'].to(device).float()[None]))
        R = len(views)
        for j, f in enumerate(win):
            cam = feats['cam_points'][0, j][None]
            m = conf_masks[j][None]
            nf = get_multiview_scale([cam], [m])
            views.append(dict(pts3d_cam=cam / nf * m[..., None].float(),
                              vggt_tokens=feats['vggt_tokens'][:, j], pose_tokens=feats['pose_tokens'][:, j],
                              salad_tokens=feats['salad_tokens'][:, j]))
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            preds = model(views, ref_ids=list(range(R)))

        scale = float(s.reshape(-1)[0])
        l2w_pts = [transform_pts(P0, preds[R + j]['pts3d_in_other_view'][0].double() * scale).float() for j in range(V)]
        l2w_conf = [preds[R + j]['conf'][0].float() for j in range(V)]
        src_all = feats['world_points'][0]
        m = conf_masks & torch.isfinite(src_all).all(-1) & torch.stack([torch.isfinite(p).all(-1) for p in l2w_pts])
        sol = ransac_sim3(src_all[m], torch.stack(l2w_pts)[m], torch.stack(l2w_conf)[m])
        inl = torch.zeros_like(m)
        inl[m] = sol['inliers']
        inlier_conf = float(torch.stack(l2w_conf)[inl].mean()) if inl.any() else 0.0
        gate_ok = bool(torch.isfinite(sol['s'])) and float(sol['s']) > 0 and \
            sol['inlier_ratio'] >= CFG['min_inlier_ratio'] and inlier_conf >= CFG['min_inlier_conf']
        anchor_sol, agree = None, True
        if anchor:
            am = top_ratio_mask(anchor['conf'], 0.5) & torch.isfinite(anchor['world_points']).all(-1)
            a_dst = anchor['entry']['pts'].to(device)
            am &= torch.isfinite(a_dst).all(-1)
            if int(am.sum()) > 100:
                a_s, a_R, a_t, a_r = irls_sim3(anchor['world_points'][am], a_dst[am])
                a_size = (a_dst[am] - a_dst[am].mean(0)).norm(dim=-1).median().clamp(min=1e-9)
                anchor_sol = dict(s=a_s, R=a_R, t=a_t, rel_residual=float(a_r.median() / a_size))
                if gate_ok:
                    rot = float(torch.rad2deg(torch.arccos(((sol['R'].T @ a_R).trace().clamp(-1, 3) - 1) / 2)))
                    agree = rot <= CFG['anchor_rot_tol'] and abs(float(sol['s'] / a_s) - 1) <= CFG['anchor_scale_tol']
        if gate_ok and agree:
            placement, T_use = 'l2w', sol
        elif anchor_sol is not None:
            placement, T_use = 'anchor', anchor_sol
        else:
            placement = 'perframe'
        if placement != 'perframe':
            nodes[w] = (T_use['s'].cpu(), T_use['R'].cpu(), T_use['t'].cpu())
        else:
            fs, fR, ft = weighted_umeyama(src_all[m], torch.stack(l2w_pts)[m], torch.stack(l2w_conf)[m])
            nodes[w] = (fs.cpu(), fR.cpu(), ft.cpu())
        if anchor_sol is not None and anchor['entry'].get('window') is not None:
            anchors[w] = dict(k=int(anchor['entry']['window']), s=anchor_sol['s'].cpu(), R=anchor_sol['R'].cpu(),
                              t=anchor_sol['t'].cpu(), rel_residual=anchor_sol['rel_residual'])
        prim = ref_frames[0]
        if placement == 'l2w' and win[0] - prim > CFG['loop_min_gap'] and pool[prim].get('window') is not None:
            loops[w] = dict(k=int(pool[prim]['window']), s=sol['s'].cpu(), R=sol['R'].cpu(), t=sol['t'].cpu(),
                            quality=sol['inlier_ratio'] * inlier_conf)

        for j, f in enumerate(win):
            frame_low = None
            if placement != 'perframe':
                pts = apply_sim3(T_use['s'], T_use['R'], T_use['t'], feats['world_points'][0, j].double()).float()
                c2w_local = feats['vggt_c2w'][0, j].double()
                c2w = torch.eye(4, dtype=torch.float64, device=device)
                c2w[:3, :3] = T_use['R'] @ c2w_local[:3, :3]
                c2w[:3, 3] = T_use['s'] * (T_use['R'] @ c2w_local[:3, 3]) + T_use['t']
                if placement == 'anchor':
                    frame_low = bool(anchor['entry'].get('low_conf', False))
                conf = feats['depth_conf'][0, j].float()
                score = float(l2w_conf[j][inl[j]].mean()) if inl[j].any() else float(l2w_conf[j].mean())
            else:
                pts, conf = l2w_pts[j], l2w_conf[j]
                score = float(conf.mean())
                rel = pose_from_encoding(preds[R + j]['pose_enc'][0]).double()
                rel[:3, 3] *= scale
                c2w = P0 @ rel
            if f not in frames or score > frames[f]['score']:
                frames[f] = dict(c2w=c2w.cpu().numpy(), pts=pts.cpu().numpy(), conf=conf.cpu().numpy(),
                                 valid=conf_masks[j].cpu().numpy(), score=score, window=w)
            low = True if placement == 'perframe' else ((score < CFG['l2w_conf_thres']) if frame_low is None else frame_low)
            entry = frame_entry(j, pts, conf, c2w, low_conf=low)
            new_pool[f] = dict(entry, score=score)
            if j in CFG['keyframe_ids']:
                db.add_or_update(f, entry, score=score)
        prev_pool = new_pool

    run_pgo(frames, nodes, anchors, loops)
    save(args, frames, frame_rgb, rng)


def run_pgo(frames, nodes, anchors, loops):
    W = max(nodes) + 1
    T = {w: to_sim3(*nodes[w]) for w in nodes}
    edges = []
    info_a = {w: 1.0 / max(a['rel_residual'] ** 2, 1e-6) for w, a in anchors.items()}
    for w, a in anchors.items():
        edges.append(dict(i=a['k'], j=w, Z=from_sim3(T[a['k']].Inv() @ to_sim3(a['s'], a['R'], a['t'])), info=info_a[w]))
    if loops:
        lam = CFG['loop_rel_weight'] * float(np.median(list(info_a.values()))) / \
            float(np.median([l['quality'] for l in loops.values()]))
        for w, l in loops.items():
            edges.append(dict(i=l['k'], j=w, Z=from_sim3(T[l['k']].Inv() @ to_sim3(l['s'], l['R'], l['t'])),
                              info=lam * l['quality']))
    init = [nodes[w] for w in range(W)]
    out = optimize_pose_graph(init, edges)
    corr = {w: correction(init[w], out[w]) for w in range(W)}
    for fr in frames.values():
        cs, cR, ct = [x.numpy() for x in corr[fr['window']]]
        fr['pts'] = (float(cs) * fr['pts'].reshape(-1, 3) @ cR.T + ct).reshape(fr['pts'].shape).astype(np.float32)
        c2w = fr['c2w'].copy()
        c2w[:3, :3] = cR @ fr['c2w'][:3, :3]
        c2w[:3, 3] = float(cs) * cR @ fr['c2w'][:3, 3] + ct
        fr['c2w'] = c2w


def save(args, frames, frame_rgb, rng):
    ids = sorted(frames)
    c2w = np.stack([frames[i]['c2w'] for i in ids]).astype(np.float64)
    np.savetxt(join(args.out_dir, 'traj_c2w.txt'), c2w.reshape(-1, 16))
    budget = max(1, CFG['num_points'] // len(ids))
    P, C = [], []
    for i in ids:
        fr = frames[i]
        p, c, v = fr['pts'].reshape(-1, 3), fr['conf'].reshape(-1), fr['valid'].reshape(-1)
        finite = np.isfinite(p).all(1)
        m = finite & v if (finite & v).any() else finite
        thr = np.quantile(c[m], 1.0 - CFG['ply_keep_ratio']) if m.any() else np.inf
        idx = np.flatnonzero(m & (c >= thr))
        if len(idx) > budget:
            idx = rng.choice(idx, budget, replace=False)
        P.append(p[idx])
        C.append(frame_rgb[i].reshape(-1, 3)[idx])
    P, C = np.concatenate(P), np.concatenate(C)
    trimesh.points.PointCloud(P, colors=np.concatenate([C, np.full((len(C), 1), 255, np.uint8)], 1)).export(
        join(args.out_dir, f'{args.name}_recon.ply'))


if __name__ == '__main__':
    main(get_parser().parse_args())
