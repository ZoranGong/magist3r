<p align="center">
  <h2 align="center">[ECCV2026] MAGiSt3R: Multi-Agent Feed-forward 3D Reconstruction from Monocular RGB Videos</h2>
 <p align="center">
    <a href="https://zorangong.github.io/">Gong Ziren</a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=1P5NDzQAAAAJ">Xiaohan Li</a>
    ·
    <a href="https://fabiotosi92.github.io/">Tosi Fabio</a>
    ·
    <a href="https://openreview.net/profile?id=%7ENinghui_Xu1">Ninghui Xu</a>
    ·
    <a href="https://stefanomattoccia.github.io/">Stefano Mattoccia</a>
    ·
    <a href="https://jianfei-cai.github.io/">Jianfei Cai</a>
    ·
    <a href="https://mattpoggi.github.io/">Matteo Poggi</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2607.15211">Paper</a> | <a href="https://zorangong.github.io/magist3r_page/">Project Page</a> </h3>
  <div align="center"></div>
</p>


## Videos

**Apart-0** Multi-agent Merging

https://github.com/user-attachments/assets/a4aab1db-3379-4322-aeb6-3b13549cca80

**Apart-1** Multi-agent Merging

https://github.com/user-attachments/assets/c731a77c-0db2-45a5-9351-e53ab4db11a0

**Apart-2** Multi-agent Merging

https://github.com/user-attachments/assets/ba663953-68c4-4292-a2fb-48c87b967b5d

## Installation
```bash
git clone --recursive https://github.com/ZoranGong/magist3r.git
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
