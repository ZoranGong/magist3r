# MAGiST3R

## Videos

**Apart-0**

<video src="VIDEO_URL_APART_0" controls muted width="100%"></video>

**Apart-1**

<video src="VIDEO_URL_APART_1" controls muted width="100%"></video>

**Apart-2**

<video src="VIDEO_URL_APART_2" controls muted width="100%"></video>

## Installation
```bash
git clone --recursive <repo>
pip install -r requirements.txt
```

## Weights
Download the weights from [Google Drive](GOOGLE_DRIVE_LINK) and place them in `weights/`:
```
weights/
├── magist3r.pth
├── vggt_omega_1b_416_reproduce.pt
└── dino_salad_2048_64.ckpt
```

## Data
Place ReplicaMultiagent in `data/ReplicaMultiagent/<group>/<sequence>/results/` and AriaMultiagent in
`data/AriaMultiagent/<room>/<agent>/results/` (RGB frames as `*.jpg`).

## Demo
```bash
bash scripts/run_ReplicaMultiagent.sh
bash scripts/run_AriaMultiagent.sh
```
Each agent writes `traj_c2w.txt` (one flattened 4x4 camera-to-world matrix per processed frame) and `<agent>_recon.ply` to `results/`.

## License
CC BY-NC-SA 4.0, see [LICENSE](LICENSE).
