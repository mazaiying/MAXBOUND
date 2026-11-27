import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import audit_exact_topk
from src.oracle import exact_maxsim_oracle
from src.maxbound_engine import MaxBoundIndex
from src.baselines.baselines import (
    B1TorchPaddedGemm,
    B2CuBLASUnpaddedExact,
    B3QuantizedDecompressGemm,
    B4PlaidGPU,
    B5MuveraGPU,
    B6CpuExactMaxSim,
)


def main():
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = np.random.default_rng(2027)
    lens = rng.integers(16, 160, size=400)
    d_offs = np.zeros(401, dtype=np.int64); np.cumsum(lens, out=d_offs[1:])
    x = rng.standard_normal((int(d_offs[-1]), 128)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    d_toks = torch.from_numpy(x).half()
    q_offs = np.arange(0, 17 * 32, 32, dtype=np.int64)
    qx = rng.standard_normal((int(q_offs[-1]), 128)).astype(np.float32)
    qx /= np.linalg.norm(qx, axis=1, keepdims=True)
    q_toks = torch.from_numpy(qx).half()

    o_ids, o_sc = exact_maxsim_oracle(q_toks.float(), q_offs, d_toks, d_offs, topk=10, device=dev)

    b1 = B1TorchPaddedGemm(d_toks, d_offs, device=dev)
    i1, s1 = b1.search(q_toks, q_offs, topk=10)
    assert audit_exact_topk(i1, s1, o_ids, o_sc)["agreement"] == 1.0
    print("[PASS] B1TorchPaddedGemm")

    b2 = B2CuBLASUnpaddedExact(d_toks, d_offs, device=dev)
    i2, s2 = b2.search(q_toks, q_offs, topk=10)
    assert audit_exact_topk(i2, s2, o_ids, o_sc)["agreement"] == 1.0
    print("[PASS] B2CuBLASUnpaddedExact")

    b3 = B3QuantizedDecompressGemm(d_toks, d_offs, device=dev)
    i3, s3 = b3.search(q_toks, q_offs, topk=10)
    assert i3.shape == (16, 10)
    print(f"[PASS] B3QuantizedDecompressGemm (approx agree={audit_exact_topk(i3, s3, o_ids, o_sc)['agreement']:.2f})")

    b4 = B4PlaidGPU(d_toks, d_offs, device=dev, n_centroids=128)
    c4 = b4.get_candidates(q_toks, q_offs, nprobe=8, ncands=100)
    i4, s4 = b4.search(q_toks, q_offs, nprobe=8, ncands=100, topk=10)
    assert i4.shape == (16, 10) and c4.shape == (16, 100)
    print(f"[PASS] B4PlaidGPU (ncands=100 agree={audit_exact_topk(i4, s4, o_ids, o_sc)['agreement']:.2f})")

    b5 = B5MuveraGPU(d_toks, d_offs, device=dev, r_reps=4, k_sim=3, d_proj=16)
    c5 = b5.get_candidates(q_toks, q_offs, ncands=100)
    i5, s5 = b5.search(q_toks, q_offs, ncands=100, topk=10)
    assert i5.shape == (16, 10) and c5.shape == (16, 100)
    print(f"[PASS] B5MuveraGPU (ncands=100 agree={audit_exact_topk(i5, s5, o_ids, o_sc)['agreement']:.2f})")

    b6 = B6CpuExactMaxSim(d_toks, d_offs)
    i6, s6 = b6.search(q_toks, q_offs, topk=10)
    assert audit_exact_topk(i6, s6, o_ids, o_sc)["agreement"] == 1.0
    print("[PASS] B6CpuExactMaxSim (48-thread OpenMP+SIMD)")

    # Test D=1024 general branch (for BGE-M3)
    d1024 = rng.standard_normal((int(d_offs[-1]), 1024)).astype(np.float32)
    d1024 /= np.linalg.norm(d1024, axis=1, keepdims=True)
    q1024 = rng.standard_normal((int(q_offs[-1]), 1024)).astype(np.float32)
    q1024 /= np.linalg.norm(q1024, axis=1, keepdims=True)
    dt1024 = torch.from_numpy(d1024).half()
    qt1024 = torch.from_numpy(q1024).half()
    o1024_i, o1024_s = exact_maxsim_oracle(qt1024.float(), q_offs, dt1024, d_offs, topk=10, device=dev)
    idx1024 = MaxBoundIndex(dt1024, d_offs, device=dev, mode="int8")
    p1024_i, p1024_s, _ = idx1024.search(qt1024, q_offs, topk=10, c_warm=16)
    assert audit_exact_topk(p1024_i, p1024_s, o1024_i, o1024_s, tol=5e-3)["agreement"] == 1.0
    print("[PASS] D=1024 BGE-M3 exactness")

    print("\nALL BASELINE + D=1024 UNIT TESTS PASSED!")


if __name__ == "__main__":
    main()
