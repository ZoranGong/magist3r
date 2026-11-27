import os.path as osp
import sys

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
VGGT_OMEGA_DIR = osp.join(ROOT_DIR, 'submodule', 'vggt-omega')
SALAD_DIR = osp.join(ROOT_DIR, 'submodule', 'salad')
if VGGT_OMEGA_DIR not in sys.path:
    sys.path.insert(0, VGGT_OMEGA_DIR)

NUM_POSE_TOKENS = 17
L2W_GRID = 14
L2W_RES = 224
