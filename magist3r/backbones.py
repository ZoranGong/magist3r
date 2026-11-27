import functools
import importlib.util
import os.path as osp
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import L2W_GRID, L2W_RES, NUM_POSE_TOKENS, SALAD_DIR
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def depth_to_cam_points(depth, intrinsic):
    N, H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                          torch.arange(W, device=depth.device, dtype=depth.dtype), indexing='ij')
    fx, fy = intrinsic[:, 0, 0][:, None, None], intrinsic[:, 1, 1][:, None, None]
    cx, cy = intrinsic[:, 0, 2][:, None, None], intrinsic[:, 1, 2][:, None, None]
    return torch.stack([(x[None] - cx) / fx * depth, (y[None] - cy) / fy * depth, depth], dim=-1)


def cam_points_to_world(cam_points, extrinsic):
    R, t = extrinsic[:, :3, :3], extrinsic[:, :3, 3]
    return torch.einsum('nij,nhwj->nhwi', R.transpose(1, 2), cam_points - t[:, None, None, :])


def extrinsic_to_c2w(extrinsic):
    w2c = torch.zeros(*extrinsic.shape[:-2], 4, 4, device=extrinsic.device, dtype=extrinsic.dtype)
    w2c[..., :3, :] = extrinsic
    w2c[..., 3, 3] = 1.0
    return torch.linalg.inv(w2c)


def tokens_to_grid(tokens, grid_hw, out=L2W_GRID):
    N, L, C = tokens.shape
    h, w = grid_hw
    grid = tokens.transpose(1, 2).reshape(N, C, h, w)
    if (h, w) != (out, out):
        grid = F.interpolate(grid, size=(out, out), mode='bilinear', align_corners=False, antialias=(h > out or w > out))
    return grid.flatten(2).transpose(1, 2).contiguous()


def resize_maps(x, size=L2W_RES, mode='bilinear'):
    squeeze = x.ndim == 3
    t = x[:, None] if squeeze else x.permute(0, 3, 1, 2)
    if t.shape[-2:] != (size, size):
        if mode == 'nearest':
            t = F.interpolate(t, size=(size, size), mode='nearest')
        else:
            t = F.interpolate(t, size=(size, size), mode=mode, align_corners=False, antialias=True)
    return t[:, 0] if squeeze else t.permute(0, 2, 3, 1).contiguous()


class VGGTOmegaWrapper(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.model = VGGTOmega()
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model'] if 'model' in state else state, strict=True)
        self.model.eval().requires_grad_(False)
        self.patch_size = 16

    @torch.no_grad()
    def forward(self, images):
        m = self.model
        B, S, _, H, W = images.shape
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=images.is_cuda):
            tokens_list, patch_start = m.aggregator(images)
        final = tokens_list[-1]
        with torch.autocast(device_type='cuda', enabled=False):
            pose_enc = m.camera_head(tokens_list, patch_token_start=patch_start)
            depth, depth_conf = m.dense_head(tokens_list, images=images, patch_token_start=patch_start)
        depth = depth[..., 0] if depth.ndim == 5 else depth
        extrinsic, intrinsic = encoding_to_camera(pose_enc, (H, W))
        cam_pts = depth_to_cam_points(depth.reshape(B * S, H, W), intrinsic.reshape(B * S, 3, 3))
        world_pts = cam_points_to_world(cam_pts, extrinsic.reshape(B * S, 3, 4))
        assert patch_start == NUM_POSE_TOKENS
        return dict(depth_conf=depth_conf.float(), pose_enc=pose_enc.float(), extrinsic=extrinsic.float(),
                    cam_points=cam_pts.reshape(B, S, H, W, 3).float(),
                    world_points=world_pts.reshape(B, S, H, W, 3).float(),
                    patch_tokens=final[:, :, patch_start:].float(),
                    patch_grid=(H // self.patch_size, W // self.patch_size),
                    pose_tokens=final[:, :, :patch_start].float())


class SaladWrapper(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        warnings.filterwarnings('ignore')
        torch.hub.load = functools.partial(torch.hub.load, verbose=False)
        dinov2 = _load_module('salad_dinov2', osp.join(SALAD_DIR, 'models', 'backbones', 'dinov2.py'))
        salad = _load_module('salad_aggregator', osp.join(SALAD_DIR, 'models', 'aggregators', 'salad.py'))
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        bb = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}
        agg = {k[len('aggregator.'):]: v for k, v in state.items() if k.startswith('aggregator.')}
        self.backbone = dinov2.DINOv2(model_name='dinov2_vitb14', num_trainable_blocks=4, norm_layer=True, return_token=True)
        self.aggregator = salad.SALAD(num_channels=agg['token_features.0.weight'].shape[1],
                                      num_clusters=agg['score.3.weight'].shape[0],
                                      cluster_dim=agg['cluster_features.3.weight'].shape[0],
                                      token_dim=agg['token_features.2.weight'].shape[0])
        self.backbone.load_state_dict(bb, strict=True)
        self.aggregator.load_state_dict(agg, strict=True)
        self.eval().requires_grad_(False)
        self.resolution = 322
        self.register_buffer('mean', torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, images):
        x = F.interpolate(images, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False,
                          antialias=True)
        feat, cls_token = self.backbone((x - self.mean) / self.std)
        desc = F.normalize(self.aggregator((feat, cls_token)).float(), dim=-1)
        return dict(spatial_tokens=feat.flatten(2).transpose(1, 2).float(), patch_grid=feat.shape[-2:],
                    global_descriptor=desc)


class Backbones(nn.Module):
    def __init__(self, vggt_ckpt, salad_ckpt):
        super().__init__()
        self.vggt = VGGTOmegaWrapper(vggt_ckpt)
        self.salad = SaladWrapper(salad_ckpt)

    @torch.no_grad()
    def forward(self, images_hr, need_salad=True):
        B, V = images_hr.shape[:2]
        out = self.vggt(images_hr)
        res = dict(
            vggt_tokens=tokens_to_grid(out['patch_tokens'].flatten(0, 1), out['patch_grid']).view(B, V, L2W_GRID ** 2, -1),
            pose_tokens=out['pose_tokens'],
            cam_points=resize_maps(out['cam_points'].flatten(0, 1), mode='nearest').view(B, V, L2W_RES, L2W_RES, 3),
            world_points=resize_maps(out['world_points'].flatten(0, 1), mode='nearest').view(B, V, L2W_RES, L2W_RES, 3),
            depth_conf=resize_maps(out['depth_conf'].flatten(0, 1)).view(B, V, L2W_RES, L2W_RES),
            vggt_c2w=extrinsic_to_c2w(out['extrinsic']),
            pose_enc=out['pose_enc'])
        if need_salad:
            s = self.salad(images_hr.flatten(0, 1))
            res['salad_tokens'] = tokens_to_grid(s['spatial_tokens'], s['patch_grid']).view(B, V, L2W_GRID ** 2, -1)
            res['global_descriptor'] = s['global_descriptor'].view(B, V, -1)
        return res
