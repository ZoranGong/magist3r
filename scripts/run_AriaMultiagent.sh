#!/bin/bash
set -e
cd "$(dirname "$0")/.."

DATA_ROOT=data/AriaMultiagent
OUT_DIR=results/AriaMultiagent
SCENES="room0/agent_0 room0/agent_1 room0/agent_2 room1/agent_0 room1/agent_1 room1/agent_2"

for S in ${SCENES}; do
    python recon.py --img_dir "${DATA_ROOT}/${S}/results" --out_dir "${OUT_DIR}/${S}" --name "$(basename "${S}")"
done
