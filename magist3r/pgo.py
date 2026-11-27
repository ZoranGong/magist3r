import numpy as np
import pypose as pp
import torch
import torch.nn as nn

ITERS = 20
CAUCHY_DELTA = 2.0
PRIOR_WEIGHT = 1e4


def to_sim3(s, R, t):
    R = torch.as_tensor(R, dtype=torch.float64)
    t = torch.as_tensor(t, dtype=torch.float64)
    return pp.Sim3(torch.cat([t.reshape(3), pp.mat2SO3(R).tensor(), torch.as_tensor([float(s)], dtype=torch.float64)]))


def from_sim3(T):
    m = T.matrix()
    sR, t = m[:3, :3], m[:3, 3]
    s = torch.det(sR).abs() ** (1.0 / 3.0)
    return s, sR / s, t


class _Graph(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.nodes = pp.Parameter(init)

    def forward(self, i, j, Z, sqrt_w, prior_T):
        T = self.nodes
        err = (Z.Inv() @ T[i].Inv() @ T[j]).Log().tensor() * sqrt_w[:, None]
        prior = (prior_T.Inv() @ T[0]).Log().tensor() * PRIOR_WEIGHT
        return torch.cat([err, prior.reshape(1, -1)], dim=0)


def optimize_pose_graph(init_nodes, edges):
    init = pp.Sim3(torch.stack([to_sim3(*n).tensor() for n in init_nodes]))
    i = torch.tensor([e['i'] for e in edges])
    j = torch.tensor([e['j'] for e in edges])
    Z = pp.Sim3(torch.stack([to_sim3(*e['Z']).tensor() for e in edges]))
    sw = torch.tensor([float(np.sqrt(e['info'])) for e in edges], dtype=torch.float64)
    model = _Graph(init.clone())
    prior_T = pp.Sim3(init[0].tensor().clone())
    opt = pp.optim.LM(model, kernel=pp.optim.kernel.Cauchy(CAUCHY_DELTA), strategy=pp.optim.strategy.TrustRegion(radius=1e3))
    prev = None
    for _ in range(ITERS):
        loss = float(opt.step(input=(i, j, Z, sw, prior_T)))
        if prev is not None and abs(prev - loss) < 1e-10 * max(1.0, prev):
            break
        prev = loss
    return [from_sim3(pp.Sim3(model.nodes[k].tensor().detach())) for k in range(len(init_nodes))]


def correction(T_old, T_new):
    return from_sim3(to_sim3(*T_new) @ to_sim3(*T_old).Inv())
