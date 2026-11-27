#!/bin/bash
set -e
cd "$(dirname "$0")/.."

DATA_ROOT=data/ReplicaMultiagent
OUT_DIR=results/ReplicaMultiagent
SCENES="Apart-0/apart_0_part1 Apart-0/apart_0_part2 Apart-1/apart_1_part1 Apart-1/apart_1_part2
        Apart-2/apart_2_part1 Apart-2/apart_2_part2 Office-0/office_0_part1 Office-0/office_0_part2"

for S in ${SCENES}; do
    python recon.py --img_dir "${DATA_ROOT}/${S}/results" --out_dir "${OUT_DIR}/${S}" --name "$(basename "${S}")"
done
