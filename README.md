

# MAGiST3R

## Videos

**Apart-0** Multi-agent Merging

<video src="https://github.com/user-attachments/assets/a4aab1db-3379-4322-aeb6-3b13549cca80" controls muted width="100%"></video>

**Apart-1** Multi-agent Merging

<video src="https://github.com/user-attachments/assets/c731a77c-0db2-45a5-9351-e53ab4db11a0" controls muted width="100%"></video>

**Apart-2** Multi-agent Merging

<video src="https://github.com/user-attachments/assets/ba663953-68c4-4292-a2fb-48c87b967b5d" controls muted width="100%"></video>

## Installation
```bash
git clone --recursive <repo>
pip install -r requirements.txt
```

## Weights
Download the weights from [Google Drive](https://drive.google.com/drive/folders/1ed9KJUkVxSjvyTcDx8AIXFbcLVyvd2nB?usp=drive_link) and place them in `weights/`:
```
weights/
├── magist3r.pth
├── vggt_omega_1b_416_reproduce.pt
└── dino_salad_2048_64.ckpt
```

## Data
Place ReplicaMultiagent in `data/ReplicaMultiagent/<group>/<sequence>/results/` and AriaMultiagent in
`data/AriaMultiagent/<room>/<agent>/results/` (RGB frames as `*.jpg`).

## Run
You can test on ReplicaMultiagent and AriaMultiagent datasets. We will soon release the multi-agent settings together with our new version on more challenging scenes. 
```bash
bash scripts/run_ReplicaMultiagent.sh
bash scripts/run_AriaMultiagent.sh
```
Each agent writes `traj_c2w.txt` (one flattened 4x4 camera-to-world matrix per processed frame) and `<agent>_recon.ply` to `results/`.

## License
CC BY-NC-SA 4.0, see [LICENSE](LICENSE).
