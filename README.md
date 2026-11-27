# MAXBOUND

Exact GPU-native multi-vector (late-interaction) retrieval via hierarchical token-group pruning and skew-resilient execution.

MAXBOUND returns the exact top-k MaxSim results while pruning over 99.9% of candidates on compressed (4/6/8-bit, Hadamard-rotated) tiles using GPU Tensor Cores, followed by on-device FP16 verification.

## Layout

| Path | Content |
|---|---|
| `src/cuda/` | CUDA kernels (`maxbound_kernels.cu`) and PyTorch bindings |
| `src/maxbound_engine.py` | Rotated tile store, pruning and verification engine |
| `src/planner.py`, `src/dual_gpu.py` | Cost-based planner (mode / bit-width / offload) and heterogeneous dual-GPU scheduler |
| `src/oracle.py` | FP64 MaxSim oracle used for exactness verification |
| `src/baselines/` | Exact and approximate baselines (cuBLAS, fused FP16, PLAID/EMVB/WARP/MUVERA-style GPU ports, CPU exact) |
| `tests/` | Exactness and baseline unit tests |
| `bench/` | Experiment drivers (`run_v6_master.sh` runs the full suite) |
| `scripts/` | Dataset construction, CUDA compilation, hardware calibration |
| `figures/plot_all.py` | Regenerates all result figures from `results/*.json` |
| `results/` | Raw result files reported in the paper |

## Requirements

- Linux, NVIDIA GPU with compute capability 8.0 or higher (evaluated on A100 80 GB and RTX 4080 16 GB)
- CUDA 12.x toolkit, a C++17 compiler, Python 3.10+
- `pip install -r requirements.txt` (PyTorch with CUDA support)
- Optional: Nsight Compute (`ncu`) for the hardware-counter experiment

## Quick start

The scripts assume the repository is located at `~/maxbound` (clone it there or create a symlink).

```bash
source env.sh
python scripts/compile_cuda.py          # builds the CUDA extension for sm_80 and sm_89 (edit the list for other GPUs)
python tests/test_exactness.py 1        # exactness check against the FP64 oracle
```

## Datasets

`scripts/build_datasets.py` and `scripts/build_msmarco_2m.py` download BEIR / MS MARCO text and encode it with ColBERTv2 (d=128) and BGE-M3 (d=1024). Place the ColBERTv2 checkpoint in `~/maxbound/colbertv2` and set `BGE_M3_DIR` to a local BGE-M3 snapshot (default: `BAAI/bge-m3` from the Hugging Face cache). Encoded data is written to `~/maxbound/data_mv/` (about 89 GB in total with the 2M-passage MS MARCO collection).

## Reproducing the experiments

```bash
source env.sh
bash bench/run_v6_master.sh     # builds data, runs all experiments, regenerates figures
```

Individual experiments: `bench/run_e0.py` (motivation and bound audit), `bench/run_e1_e2.py`, `bench/run_e3_e4.py`, `bench/run_e5_e6_e7.py`, `bench/run_v5_suite.py`, `bench/run_e12_e13.py`. Every result is written to `results/*.json` together with the random seed (2027) and timing statistics (3 warmup and 15 timed iterations, CUDA events).

## License

MIT, see `LICENSE`.
