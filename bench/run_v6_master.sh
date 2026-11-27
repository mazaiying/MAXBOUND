#!/usr/bin/env bash
set -euo pipefail
cd ~/maxbound
source ~/maxbound/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results figures

rm -f ~/maxbound/*.tar.gz ~/maxbound/*.tgz ~/maxbound/check_fast.py ~/maxbound/verify_v6.py
echo "=== [$(date)] Disk before Stage A ==="
du -sh ~/maxbound

echo "=== [$(date)] Stage A: Building msmarco_2m (2,000,000 official MS MARCO passages, ~130M tokens) across A100 + RTX 4080 ==="
python -u scripts/build_msmarco_2m.py
echo "=== [$(date)] Disk after building msmarco_2m and cleaning raw_msmarco ==="
du -sh ~/maxbound ~/maxbound/data_mv/msmarco_2m

echo "=== [$(date)] Stage B: Running E7 Zero-Leakage Synthetic Calibration + Heuristic Comparison ==="
python -u bench/run_v5_suite.py e7

echo "=== [$(date)] Stage C: Running E0 + E0B (12.4M-pair Slack Audit) + E0C (Rare-Entity Exactness Value) ==="
python -u bench/run_e0.py 1

echo "=== [$(date)] Stage D: Running E5 Skew & Query-Length Robustness ==="
python -u -c "from bench.run_e5_e6_e7 import run_e5; run_e5(1)"

echo "=== [$(date)] Stage E: Running E3 Constrained-VRAM Scaling (incl. 130M-token msmarco_2m on 16GB RTX 4080) ==="
python -u bench/run_e3_e4.py

echo "=== [$(date)] Stage F: Running E1 (12 Datasets), E2 (Pareto + EMVB + WARP), E4 (5-Dataset DualGPU), E6, E8, E10, E11, E14 ==="
python -u bench/run_v5_suite.py e1 e2 e4 e6 e8_10_11 e14

echo "=== [$(date)] Stage G: Running E12 (Nsight Compute PMU), E13 (Dynamic + Dense B2 Append), and E9 Consolidated FP64 Audit ==="
python -u bench/run_e12_e13.py 1

echo "=== [$(date)] Stage H: Plotting all 10 publication figures (fig1..fig10) ==="
python -u figures/plot_all.py

echo "=== [$(date)] Final Disk Footprint Check (< 100 GB strict budget) ==="
du -sh ~/maxbound
echo "=== [$(date)] ALL V6 EXPERIMENTS AND FIGURES COMPLETED 100% ==="
