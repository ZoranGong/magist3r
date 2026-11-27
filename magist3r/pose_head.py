import torch.nn as nn

from . import NUM_POSE_TOKENS  # noqa: F401
from vggt_omega.models.heads import CameraHead


class MagmaPoseHead(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.in_proj = nn.Linear(dim_in, 2048)
        self.camera_head = CameraHead(dim_in=2048)

    def forward(self, pose_groups):
        x = self.in_proj(pose_groups.float())
        return self.camera_head([x], patch_token_start=x.shape[2])
